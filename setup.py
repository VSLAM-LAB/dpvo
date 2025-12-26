import os.path as osp
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))
conda_prefix = os.environ.get("PREFIX", os.environ.get("CONDA_PREFIX", ""))
eigen_path = osp.join(conda_prefix, 'include', 'eigen3')

setup(
    name='dpvo',
    packages=find_packages(),
    py_modules=['vslamlab_dpvo_mono'],
    entry_points={
        'console_scripts': [
            'vslamlab_dpvo_mono = vslamlab_dpvo_mono:main',
        ]
    },
    ext_modules=[
        CUDAExtension('cuda_corr',
            sources=['dpvo/altcorr/correlation.cpp', 'dpvo/altcorr/correlation_kernel.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            }),
        CUDAExtension('cuda_ba',
            sources=['dpvo/fastba/ba.cpp', 'dpvo/fastba/ba_cuda.cu', 'dpvo/fastba/block_e.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            },
            include_dirs=[
                eigen_path
                ]
            ),
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'dpvo/lietorch/include'), 
                eigen_path
            ],
            sources=[
                'dpvo/lietorch/src/lietorch.cpp', 
                'dpvo/lietorch/src/lietorch_gpu.cu',
                'dpvo/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3'],}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
