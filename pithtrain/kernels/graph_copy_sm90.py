"""
SM90 pointer-table transfers; all addresses are runtime device-table values.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import ld_b32_coherent, ld_v4, st_b32, st_v4

__all__ = ["graph_copy_sm90"]


class GraphCopy:
    def __init__(self, counts, base, into_static):
        self.counts, self.base, self.into_static = counts, base, into_static
        self.begins = tuple(sum((n + 1023) // 1024 for n in counts[:i]) for i in range(len(counts)))
        self.blocks = sum((n + 1023) // 1024 for n in counts)

    @cute.jit
    def __call__(self, table: cute.Tensor, fixed: cute.Tensor, stream):
        self.kernel(table, fixed).launch(grid=[self.blocks, 1, 1], block=[256, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, table: cute.Tensor, fixed: cute.Tensor):
        tidx, unused_y, unused_z = cute.arch.thread_idx()
        block, unused_by, unused_bz = cute.arch.block_idx()
        for i in cutlass.range_constexpr(len(self.counts)):
            begin = self.begins[i]
            if block >= begin and block < begin + (self.counts[i] + 1023) // 1024:
                dynamic, static = Int64(table[self.base + i]), Int64(fixed[i])
                if dynamic != static:
                    src, dst = (dynamic, static) if cutlass.const_expr(self.into_static) else (static, dynamic)
                    offset = (Int64(block) - begin) * 1024 + Int64(tidx) * 4
                    if (src % 16 == 0) and (dst % 16 == 0) and (offset + 3 < self.counts[i]):
                        values = ld_v4(src + Int64(offset) * 4, Int32(1))
                        st_v4(dst + Int64(offset) * 4, values, Int32(1))
                    else:
                        for lane in cutlass.range_constexpr(4):
                            valid = Int32(offset + lane < self.counts[i])
                            value = ld_b32_coherent(src + Int64(offset + lane) * 4, valid, Int32(0))
                            st_b32(dst + Int64(offset + lane) * 4, Uint32(value), valid)


@cache
def compiled_copy(counts, base, into_static):
    return cute.compile(GraphCopy(counts, base, into_static), fake(Int64, (cute.sym_int(),), (1,), 8), fake(Int64, (len(counts),), (1,), 8), fake_stream(), options="--enable-tvm-ffi")


def graph_copy_sm90(table, fixed_table, word_counts, base, into_static):
    """
    SM90 implementation of the pointer-table contract in graph_copy.
    """
    if any(word_counts):
        compiled_copy(tuple(word_counts), base, into_static)(table, fixed_table)
