"""Pointer-table transfer kernels preserve physical bytes on every supported GPU."""

import pytest
import torch

from pithtrain.kernels import graph_copy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("shape", [(0,), (3, 132), (1, 256, 2048), (1, 256, 2880)])
@pytest.mark.parametrize("into_static", [False, True])
def test_graph_copy_bits(shape, into_static):
    torch.manual_seed(93)
    count = 1
    for dim in shape:
        count *= dim
    bits = torch.randint(-(2**31), 2**31 - 1, (count + 2,), device="cuda", dtype=torch.int32)
    if count >= 6:
        bits[2:8] = torch.tensor(
            [0, -2147483648, 2139095040, -8388608, 2143289345, 2139095041],
            dtype=torch.int32,
            device="cuda",
        )
    sources = [
        bits[2:].view(torch.float32).view(shape),
        torch.randint(0, 256, (132,), device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn),
        torch.arange(66, device="cuda", dtype=torch.bfloat16)[2:].view(8, 8).T,
        torch.tensor(2**40 + 7, device="cuda", dtype=torch.int64),
    ]
    destinations = [
        torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device) for x in sources
    ]
    static = destinations if into_static else sources
    for update in range(3):
        if update:
            for source in sources:
                source.as_strided((source.numel(),), (1,)).view(torch.int32).add_(17)
            sources = [x.clone(memory_format=torch.preserve_format) for x in sources]
            destinations = (
                [
                    torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
                    for x in sources
                ]
                if not into_static
                else destinations
            )
            if not into_static:
                sources = static
        dynamic = sources if into_static else destinations
        table = torch.tensor(
            [0] + [x.data_ptr() for x in dynamic], dtype=torch.int64, device="cuda"
        )
        fixed = torch.tensor([x.data_ptr() for x in static], dtype=torch.int64, device="cuda")
        counts = tuple(x.numel() * x.element_size() // 4 for x in static)
        graph_copy(table, fixed, counts, base=1, into_static=into_static)
        for actual, expected in zip(destinations, sources):
            a = actual.as_strided((actual.numel(),), (1,)).view(torch.uint8)
            b = expected.as_strided((expected.numel(),), (1,)).view(torch.uint8)
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("into_static", [False, True])
def test_graph_copy_large_word_offset(into_static):
    # Each operand crosses the signed 32-bit word-index boundary, including a partial tile.
    count = 2**31 + 1027
    if torch.cuda.mem_get_info()[0] < 20 * 2**30:
        pytest.skip("Large-offset regression requires 20 GiB free GPU memory")
    source = torch.full((count,), -156712345, device="cuda", dtype=torch.int32)
    source[-1027:] = torch.arange(1027, device="cuda", dtype=torch.int32)
    destination = torch.zeros_like(source)
    static, dynamic = (destination, source) if into_static else (source, destination)
    table = torch.tensor([dynamic.data_ptr()], device="cuda", dtype=torch.int64)
    fixed = torch.tensor([static.data_ptr()], device="cuda", dtype=torch.int64)
    graph_copy(table, fixed, (count,), into_static=into_static)
    assert torch.equal(destination, source)
