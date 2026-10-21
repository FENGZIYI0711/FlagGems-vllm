# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Normalization and FP8 decoding shared by the migrated MetaX grouped GEMM."""

import torch
import triton
import triton.language as tl

# Local copies of the original normalization and FP8 decode helpers.
BIAS_NONE = 0
BIAS_VECTOR = 1
BIAS_GROUPED = 2


@triton.jit
def _decode_e4m3(bits, FNUZ: tl.constexpr):
    # Load FP8 storage as bytes: several backends cannot represent FP8 pointers.
    bits = bits.to(tl.int32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    if FNUZ:
        power = ((exponent + 119) << 23).to(tl.float32, bitcast=True)
        subnormal = mantissa.to(tl.float32) * 0.0009765625
        is_nan = bits == 128
    else:
        power = ((exponent + 120) << 23).to(tl.float32, bitcast=True)
        subnormal = mantissa.to(tl.float32) * 0.001953125
        is_nan = (bits & 127) == 127
    value = tl.where(exponent == 0, subnormal, power * (1.0 + mantissa * 0.125))
    value = tl.where((bits & 128) != 0, -value, value)
    return tl.where(is_nan, float("nan"), value).to(tl.float16)


def _check_dims(mat_a, mat_b):
    if mat_a.dim() not in (2, 3):
        raise RuntimeError("mat_a has to be 2D or 3D")
    if mat_b.dim() not in (2, 3):
        raise RuntimeError("mat_b has to be 2D or 3D")
    if mat_a.dtype != mat_b.dtype:
        raise RuntimeError(
            f"mat_a and mat_b must have the same dtype, got {mat_a.dtype} and {mat_b.dtype}"
        )


def _check_offsets(offs, need_offsets, num_groups):
    if need_offsets:
        if offs is None:
            raise RuntimeError("offs must be provided when either input is 2D")
        if offs.dim() != 1:
            raise RuntimeError("offs has to be 1D")
        if offs.dtype != torch.int32:
            raise RuntimeError("offs has to be int32")
        if offs.numel() != num_groups:
            raise RuntimeError(
                f"offs length must match the group count, got {offs.numel()} and {num_groups}"
            )
        return offs.contiguous()

    if offs is not None:
        raise RuntimeError("offs must be None when both inputs are 3D")
    return None


def _resolve_shapes(mat_a, mat_b, offs):
    a_is_2d = mat_a.dim() == 2
    b_is_2d = mat_b.dim() == 2
    need_offsets = a_is_2d or b_is_2d

    if a_is_2d:
        M, K = mat_a.shape
    else:
        num_groups_a, M, K = mat_a.shape

    if b_is_2d:
        BK, N = mat_b.shape
    else:
        num_groups_b, BK, N = mat_b.shape

    if K != BK:
        raise RuntimeError(
            f"mat_a and mat_b shapes cannot be multiplied ({K} and {BK})"
        )

    if need_offsets:
        if a_is_2d and b_is_2d:
            num_groups = offs.numel() if offs is not None else 0
            out_shape = (num_groups, M, N)
        elif a_is_2d:
            num_groups = num_groups_b
            out_shape = (M, N)
        else:
            num_groups = num_groups_a
            out_shape = (M, N)
    else:
        if num_groups_a != num_groups_b:
            raise RuntimeError(
                f"matrix batch sizes have to match, got {num_groups_a} and {num_groups_b}"
            )
        num_groups = num_groups_a
        out_shape = (num_groups, M, N)

    offs = _check_offsets(offs, need_offsets, num_groups)
    return a_is_2d, b_is_2d, num_groups, M, N, K, out_shape, offs


def _normalize_scale(scale, mat, *, dim, num_groups, scale_multiplier, name):
    if scale.dtype != torch.float32:
        raise RuntimeError(f"{name} must be a float32 tensor")

    if mat.dim() == 2:
        expected = mat.shape[dim] * scale_multiplier
        if scale.dim() != 1 or scale.numel() != expected:
            raise RuntimeError(f"{name} must be a 1D tensor with length {expected}")
        return scale.reshape(expected).contiguous()

    expected_shape = (num_groups, mat.shape[1 + dim])
    if scale.dim() != 2 or tuple(scale.shape) != expected_shape:
        raise RuntimeError(f"{name} must have shape {expected_shape}")
    return scale.contiguous()


def _normalize_bias(bias, *, a_is_2d, b_is_2d, num_groups, N):
    if bias is None:
        return None, BIAS_NONE

    if bias.dim() == 1 and bias.numel() == N:
        return bias.contiguous(), BIAS_VECTOR

    can_use_grouped_bias = not (b_is_2d and not a_is_2d)
    if can_use_grouped_bias and bias.numel() == num_groups * N:
        return bias.reshape(num_groups, N).contiguous(), BIAS_GROUPED

    expected = f"({N},)"
    if can_use_grouped_bias:
        expected = f"{expected} or ({num_groups}, {N})"
    raise RuntimeError(f"bias must have shape {expected}")
