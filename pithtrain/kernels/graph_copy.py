"""
Architecture dispatch for pointer-table CUDA graph transfers.
"""

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["graph_copy"]


def graph_copy(table, fixed_table, word_counts, *, base=0, into_static=True):
    """
    Copy dense physical bytes between static addresses and current device-table addresses.

    table is a contiguous CUDA int64 vector. Entries starting at base hold device addresses,
    one per pair. fixed_table is a CUDA int64 vector of static addresses. word_counts
    gives each pair's size in four-byte words. Each pair must have identical dense shape, strides and dtype, four-byte
    aligned pointers and byte lengths, no conjugate/negative flags, and enough backing storage.
    Tensors may have any dtype or dense transpose. Empty and identical-pointer pairs perform
    no memory access. All other source/destination ranges must be disjoint, including shifted
    overlap and dependencies between different pairs. Read-only sources may alias each other.
    With into_static=True the table names sources; otherwise it names destinations. Copies run
    on the current stream, preserve every bit and support SM90/SM100 without architecture logic.
    The caller owns all pointed-to allocations until execution completes and handles autograd
    version increments. This raw kernel neither tracks pointer lifetimes nor updates versions.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.graph_copy_sm90 import graph_copy_sm90
            return graph_copy_sm90(table, fixed_table, word_counts, base, into_static)
        case 10:
            from pithtrain.kernels.graph_copy_sm100 import graph_copy_sm100
            return graph_copy_sm100(table, fixed_table, word_counts, base, into_static)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
