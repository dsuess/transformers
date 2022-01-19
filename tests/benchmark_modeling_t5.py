import torch
import pytest
from transformers.models.t5.modeling_t5 import T5LayerNorm

@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("device", ["cuda"])
def test_t5_layer_norm(benchmark, dtype, device):
    dtype = getattr(torch, dtype)
    x = torch.randn(16, 62, 512, dtype=dtype).to(device)
    module = T5LayerNorm(hidden_size=512).to(dtype).to(device)

    benchmark(lambda: module(x))