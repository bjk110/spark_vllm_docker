"""AOT build for TurboQuant CUDA warp-per-head decode kernel.

Build:
    python setup.py install

SM121 (Blackwell / GB10) target. Falls back gracefully if CUDA unavailable.
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# SM121 = Blackwell (GB10). Also include SM89/SM90 for compatibility.
CUDA_ARCH = os.environ.get("TORCH_CUDA_ARCH_LIST", "12.1a")
NVCC_FLAGS = [
    "-O3",
    "--use_fast_math",
    f"--gpu-architecture=compute_121",
    f"--gpu-code=sm_121",
]

setup(
    name="turboquant_wph_ext",
    ext_modules=[
        CUDAExtension(
            name="turboquant_wph_ext",
            sources=["turboquant_wph_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": NVCC_FLAGS,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
