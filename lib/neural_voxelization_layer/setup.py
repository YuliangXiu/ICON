from setuptools import setup
import os

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='voxelize_cuda',
    ext_modules=[
        CUDAExtension('voxelize_cuda', [
            os.path.join("cuda", "voxelize_cuda_kernel.cu"),
            os.path.join("cuda", "voxelize_cuda.cpp"),
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })