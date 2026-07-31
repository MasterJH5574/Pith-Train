from typing import Literal


class ModelImplMode:
    """
    Model Implementation Mode. Turn on the reference implementation to validate
    the correctness of the optimized, possibly distributed implementation.
    """

    use_reference_fwd = False
    fp8_training: Literal["deep-gemm", "disabled"] = "disabled"


def get_linear_cls():
    """Return the appropriate Linear class based on ModelImplMode.fp8_training."""
    if ModelImplMode.fp8_training == "deep-gemm":
        from pithtrain.layers.deepgemm_fp8_linear import FP8Linear

        return FP8Linear
    import transformer_engine.pytorch as te

    return te.Linear


def get_group_linear_cls():
    """Return the appropriate GroupLinear class based on ModelImplMode.fp8_training."""
    if ModelImplMode.fp8_training == "deep-gemm":
        from pithtrain.layers.deepgemm_fp8_linear import FP8GroupLinear

        return FP8GroupLinear
    from pithtrain.layers.te_group_linear import TEGroupLinear

    return TEGroupLinear
