# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm + indexed adaLN scale/shift Triton kernel for bf16 rows.

Replaces the eager two-kernel chain used by the MiniMax-H3-Image DiT blocks

    ``h = nn.RMSNorm(x)``                                (aten vectorized LN)
    ``indexed_scale_shift_bf16_(h, shift, scale, indices)``  (Triton modulate)

with a single launch while reproducing the chain bit for bit
(``torch.equal``) on ROCm/HCU, so callers need no quality gate beyond a
one-time equality check.

Numerics contract (replicates torch 2.11 ROCm
``at::native::vectorized_layer_norm_kernel<c10::BFloat16, float, true>``
followed by ``indexed_modulation_triton``):

- launch geometry: one block of 256 threads per row (``dim3(64, 4)``):
  ``thrx = threadIdx.x + threadIdx.y * 64``; thread ``t`` walks vec4
  indices ``t, t+256, ... < N/4`` and serially accumulates
  ``sigma2 = fma(val, val, sigma2)`` over the 4 elements in order
  (hipcc contracts the source ``sigma2 + val*val`` into an FMA);
- intra-warp reduction: ``__shfl_down`` offsets 32,16,8,4,2,1 over wave64
  == a halving tree (lower lane + upper lane) per warp;
- inter-warp (4 warps): offset 2 merges warps (0,2) and (1,3), offset 1
  merges the two results — again a halving tree over the 4 warp sums;
- ``sigma2 /= N`` (exact fp32 division, no constant reciprocal folding),
  ``rstd = rsqrtf(sigma2 + eps)`` — ``tl.rsqrt`` matches HIPCC ``rsqrtf``
  bit for bit (verified on gfx938; ``1 / sqrt`` does *not*);
- normalized output ``bf16(fp32(gamma) * (rstd * fp32(x)))`` — two
  separate multiplies, no FMA — rounded to bf16 precision as the eager
  kernel's tensor boundary, then the modulation chain of
  ``indexed_modulation_triton``: ``round(1 + scale)``,
  ``round(y * that)``, store ``bf16(prod + shift)``.

The aten geometry above is ROCm-specific (wave64, 256 threads); CUDA uses
128 threads / wave32 and a different fold, so this kernel is HIP-only.
Callers must verify ``torch.equal`` once at runtime and fall back on
mismatch (see ``minimax_h3.py``).
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.diffusion.common.numerics import round_bf16_to_fp32
from sglang.kernels.ops.diffusion.common.platform import is_hip


@triton.jit
def _split_halves(x, rows: tl.constexpr, half: tl.constexpr):
    # (rows, 2*half) -> two (rows, half) tensors pairing element i with
    # i+half, the tree a shfl_down fold with the largest offset first
    # produces (lower lane + upper lane).
    x = tl.reshape(x, (rows, 2, half), can_reorder=False)
    x = tl.permute(x, (0, 2, 1))
    return tl.split(x)


@triton.jit
def _fold_halves(x, rows: tl.constexpr, half: tl.constexpr):
    lo, hi = _split_halves(x, rows, half)
    return lo + hi


@triton.jit
def _rmsnorm_indexed_scale_shift_kernel(
    y_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    eps,
    D: tl.constexpr,
    PASS2_BLOCK: tl.constexpr,
):
    NVEC: tl.constexpr = D // 4
    row = tl.program_id(0).to(tl.int64)
    row_base = row * D

    # ----- pass 1: sum of squares in aten's exact thread/fold order -----
    t = tl.arange(0, 256)
    acc = tl.zeros((256,), dtype=tl.float32)
    for it in tl.static_range((NVEC + 255) // 256):
        vec_idx = it * 256 + t
        valid = vec_idx < NVEC
        cols = vec_idx[:, None] * 4 + tl.arange(0, 4)[None, :]
        x4 = tl.load(
            x_ptr + row_base + cols, mask=valid[:, None], other=0.0
        ).to(tl.float32)
        g = tl.reshape(x4, (256, 2, 2), can_reorder=False)
        p02, p13 = tl.split(g)
        e0, e2 = tl.split(tl.reshape(p02, (256, 1, 2), can_reorder=False))
        e1, e3 = tl.split(tl.reshape(p13, (256, 1, 2), can_reorder=False))
        e0 = tl.reshape(e0, (256,), can_reorder=False)
        e1 = tl.reshape(e1, (256,), can_reorder=False)
        e2 = tl.reshape(e2, (256,), can_reorder=False)
        e3 = tl.reshape(e3, (256,), can_reorder=False)
        n0 = tl.fma(e0, e0, acc)
        n1 = tl.fma(e1, e1, n0)
        n2 = tl.fma(e2, e2, n1)
        n3 = tl.fma(e3, e3, n2)
        # threads whose vector index is out of range skip the iteration
        acc = tl.where(valid, n3, acc)

    # intra-warp halving tree (wave64 shfl_down offsets 32..1): acc is
    # (256,) with warp = t // 64, lane = t % 64  ->  (4, 64)
    a = tl.reshape(acc, (4, 64), can_reorder=False)
    a = _fold_halves(a, 4, 32)
    a = _fold_halves(a, 4, 16)
    a = _fold_halves(a, 4, 8)
    a = _fold_halves(a, 4, 4)
    a = _fold_halves(a, 4, 2)
    a = _fold_halves(a, 4, 1)  # (4, 1): warp sums in warp order
    # inter-warp: w0+=w2, w1+=w3, then w0+=w1  == halving tree over 4
    v = tl.reshape(a, (1, 4), can_reorder=False)
    v = _fold_halves(v, 1, 2)
    v = _fold_halves(v, 1, 1)
    total = tl.reshape(v, (), can_reorder=False)

    rstd = tl.rsqrt(total / D + eps)  # exact div, then HIP rsqrtf

    idx = tl.load(indices_ptr + row * stride_indices).to(tl.int64)

    # ----- pass 2: normalize + modulate, in the eager rounding order -----
    for it in tl.static_range((D + PASS2_BLOCK - 1) // PASS2_BLOCK):
        cols = it * PASS2_BLOCK + tl.arange(0, PASS2_BLOCK)
        mask = cols < D
        x = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = round_bf16_to_fp32(w * (rstd * x))  # RMSNorm's bf16 tensor boundary
        sc = tl.load(
            scale_ptr + idx * stride_scale_row + cols, mask=mask, other=0.0
        ).to(tl.float32)
        sh = tl.load(
            shift_ptr + idx * stride_shift_row + cols, mask=mask, other=0.0
        ).to(tl.float32)
        one_plus = round_bf16_to_fp32(1.0 + sc)
        prod = round_bf16_to_fp32(y * one_plus)
        tl.store(y_ptr + row_base + cols, prod + sh, mask=mask)


def can_use_fused_rmsnorm_indexed_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    """Whether the fused kernel can replace the eager chain for this input.

    Mirrors the aten vectorized dispatch guards: bf16, contiguous, hidden a
    multiple of 8 (16B row alignment) and at least 1024 wide, 16B-aligned
    data pointers, plus the Triton modulate kernel's operand layout.  The
    reduction geometry is ROCm-specific, so CUDA never qualifies.
    """
    if not is_hip():
        return False
    if x.dim() != 2 or not x.is_contiguous() or x.shape[0] == 0:
        return False
    hidden = x.shape[-1]
    if hidden < 1024 or hidden % 8 != 0:
        return False
    return (
        x.dtype is torch.bfloat16
        and x.is_cuda
        and x.data_ptr() % 16 == 0
        and weight.dtype is torch.bfloat16
        and weight.is_cuda
        and weight.device == x.device
        and weight.shape == (hidden,)
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
        and shift.dtype is torch.bfloat16
        and shift.is_cuda
        and shift.device == x.device
        and shift.dim() == 2
        and shift.shape[-1] == hidden
        and shift.stride(-1) == 1
        and scale.dtype is torch.bfloat16
        and scale.is_cuda
        and scale.device == x.device
        and scale.dim() == 2
        and scale.shape[-1] == hidden
        and scale.stride(-1) == 1
        and indices.is_cuda
        and indices.device == x.device
        and indices.dim() == 1
        and indices.shape[0] == x.shape[0]
    )


def fused_rmsnorm_indexed_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
    *,
    num_warps: int = 1,
    pass2_block: int = 1024,
) -> torch.Tensor:
    """``indexed_scale_shift(RMSNorm(x))``, bit-exact vs the eager chain."""
    rows, hidden = x.shape
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _rmsnorm_indexed_scale_shift_kernel[(rows,)](
            out,
            x,
            weight,
            shift,
            scale,
            indices,
            shift.stride(0),
            scale.stride(0),
            indices.stride(0),
            eps,
            D=hidden,
            PASS2_BLOCK=pass2_block,
            num_warps=num_warps,
        )
    return out


__all__ = [
    "can_use_fused_rmsnorm_indexed_scale_shift",
    "fused_rmsnorm_indexed_scale_shift",
]
