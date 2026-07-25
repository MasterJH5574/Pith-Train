from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            name="pithtrain_ext",
            sources=["pithtrain/csrc/grouped_gemm.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
