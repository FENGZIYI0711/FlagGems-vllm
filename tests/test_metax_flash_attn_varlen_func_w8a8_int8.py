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

import pytest
import torch

import flaggems_vllm

from .test_flash_attn_varlen_func_w8a8_int8 import _run_case

pytestmark = [
    pytest.mark.flash_attn_varlen_func_w8a8_int8,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX backend"),
]


@pytest.fixture(autouse=True)
def exact_reference_matmul():
    # TF32 reference rounding can exceed the existing FP32 LSE tolerance.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("dim", [64, 96, 128, 192, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "qlens,klens",
    [
        ([512], [512]),
        ([513, 129], [513, 257]),
        ([129], [65]),
        ([128], [1024]),
        ([31, 65, 127], [63, 129, 513]),
    ],
)
def test_metax_tile_boundaries(dim, causal, dtype, qlens, klens):
    _run_case(qlens, klens, dim=dim, causal=causal, dtype=dtype)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("broadcast_scales", [False, True])
def test_metax_strided_ragged(causal, broadcast_scales):
    _run_case(
        [129, 513],
        [257, 769],
        dim=128,
        causal=causal,
        strided=True,
        broadcast_scales=broadcast_scales,
    )


@pytest.mark.parametrize("window,cap", [((31, 0), 0), ((32, 9), 3)])
def test_metax_score_modifiers(window, cap):
    _run_case([129, 513], [257, 769], window=window, cap=cap)


@pytest.mark.parametrize("dim", [8, 24, 32])
def test_metax_small_head(dim):
    _run_case([31, 65], [63, 129], dim=dim, causal=True)


def test_metax_empty_requests():
    _run_case([131, 0, 3], [1, 0, 0], causal=True)


@pytest.mark.parametrize("causal", [False, True])
def test_metax_paged_mha(causal):
    _run_case([17, 129], [145, 257], causal=causal, paged=True)


@pytest.mark.parametrize("causal", [False, True])
def test_metax_shape_changes(causal):
    from .test_flash_attn_varlen_func_w8a8_int8 import _inputs, _reference

    for batch, length in ((1, 512), (2, 1024), (4, 2048)):
        lengths = [length] * batch
        quantized, scales, references = [], [], []
        for index in range(3):
            quant, scale, _, cu = _inputs(lengths, 4, 64, broadcast_scales=True)
            torch.manual_seed(123 + index)
            quant.random_(-127, 128)
            reference = (
                quant.reshape(batch, length, 4, 64).float() * scale[:, None, :, 0, None]
            ).reshape_as(quant)
            quantized.append(quant)
            scales.append(scale)
            references.append(reference)
        result, lse = flaggems_vllm.flash_attn_varlen_func(
            *quantized,
            length,
            cu,
            length,
            cu,
            q_descale=scales[0],
            k_descale=scales[1],
            v_descale=scales[2],
            causal=causal,
            return_softmax_lse=True,
        )
        reference, reference_lse = _reference(*references, lengths, lengths, causal)
        torch.testing.assert_close(result.float(), reference, atol=0.025, rtol=0.025)
        torch.testing.assert_close(lse, reference_lse, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_metax_wide_output(dtype, causal):
    _run_case([1024, 513], [1024, 769], dim=192, dtype=dtype, causal=causal)
