"""
SM100 pointer-table transfers through TIRx-lite; fixed and dynamic addresses are runtime values.
"""

from functools import cache

import tvm_tirx
import tirx_kernels.tirx_lite as txl

__all__ = ["graph_copy_sm100"]

TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a", "backend_config": {"cuda": {"compiler": "nvcc", "target_format": "cubin"}}})
EXECUTABLES = []


@cache
def compiled_copy(counts, base, into_static):
    begins = tuple(sum((n + 1023) // 1024 for n in counts[:i]) for i in range(len(counts)))
    blocks = sum((n + 1023) // 1024 for n in counts)
    table_size = txl.Var("table_size", "int32")

    @txl.kernel()
    def copy(table: txl.Tensor((table_size,), "int64"), fixed: txl.Tensor((len(counts),), "int64"), table_size: table_size):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=blocks, block=256))
        block, tidx = txl.cta_id(), txl.thread_id()
        for i in range(len(counts)):
            with txl.if_((block >= begins[i]) & (block < begins[i] + (counts[i] + 1023) // 1024)), txl.then_():
                dynamic, static = txl.local_scalar("int64"), txl.local_scalar("int64")
                txl.ptx["ld.global.b64"](dynamic, table.ptr_to([base + i]))
                txl.ptx["ld.global.b64"](static, fixed.ptr_to([i]))
                with txl.if_(dynamic != static), txl.then_():
                    src, dst = (dynamic, static) if into_static else (static, dynamic)
                    offset = (txl.cast(block, "int64") - begins[i]) * 1024 + txl.cast(tidx, "int64") * 4
                    for lane in range(4):
                        with txl.if_(offset + lane < counts[i]), txl.then_():
                            value = txl.local_scalar("uint32")
                            txl.ptx["ld.global.b32"](value, src + txl.cast(offset + lane, "int64") * 4)
                            txl.ptx["st.global.b32"](dst + txl.cast(offset + lane, "int64") * 4, value)

    executable = copy.compile(TARGET)
    EXECUTABLES.append(executable)
    return executable["main"]


def graph_copy_sm100(table, fixed_table, word_counts, base, into_static):
    """
    SM100 implementation of the pointer-table contract in graph_copy.
    """
    if any(word_counts):
        compiled_copy(tuple(word_counts), base, into_static)(table, fixed_table, table.numel())
