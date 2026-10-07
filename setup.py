import hashlib
import os
import os.path as osp
import subprocess
import sys

import torch
from setuptools import Extension, setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))
conda_prefix = os.environ.get("PREFIX", os.environ.get("CONDA_PREFIX", ""))
eigen_path = osp.join(conda_prefix, 'include', 'eigen3')


class CMakeExtension(Extension):
    """A pybind11 module built by its own CMakeLists.txt (DPViewer, DPRetrieval)."""

    def __init__(self, name, sourcedir):
        super().__init__(name, sources=[])
        self.sourcedir = osp.join(ROOT, sourcedir)


def cmake_cuda_architectures():
    # TORCH_CUDA_ARCH_LIST ("7.5;8.6;9.0+PTX") -> CMAKE_CUDA_ARCHITECTURES ("75-real;86-real;90"); unset -> local GPU
    archs = os.environ.get("TORCH_CUDA_ARCH_LIST", "").replace(" ", ";")
    if not archs:
        return "native"
    out = []
    for arch in filter(None, archs.split(";")):
        ptx = arch.endswith("+PTX")
        number = arch.removesuffix("+PTX").replace(".", "")
        out.append(number if ptx else f"{number}-real")
    return ";".join(out)


class BuildExt(BuildExtension):
    """torch's BuildExtension for the CUDA extensions, plus CMake for the CMakeExtension ones."""

    def finalize_options(self):
        super().finalize_options()
        # One build directory per environment: pixi builds the CUDA variants in parallel from this same source tree,
        # and they must not share object files or a CMake cache (which also records the build env's tool paths)
        self.build_temp = osp.join(self.build_temp, hashlib.sha1(sys.prefix.encode()).hexdigest()[:10])

    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)

        extdir = osp.abspath(osp.dirname(self.get_ext_fullpath(ext.name)))
        build_temp = osp.join(osp.abspath(self.build_temp), ext.name)
        os.makedirs(build_temp, exist_ok=True)
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path};{conda_prefix}",
            f"-DCMAKE_CUDA_ARCHITECTURES={cmake_cuda_architectures()}",
            f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}",
            "-GNinja",
        ]
        cmake_args += [arg for arg in os.environ.get("CMAKE_ARGS", "").split(" ") if arg]
        subprocess.check_call(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--verbose"], cwd=build_temp)


# Package metadata and console scripts live in pyproject.toml; setup.py only declares packages and extensions.
setup(
    packages=find_packages() + ['dpviewer'],
    package_dir={'dpviewer': 'DPViewer/dpviewer'},
    py_modules=['vslamlab_dpvo_mono'],
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
        CMakeExtension('dpviewerx', 'DPViewer'),       # Pangolin viewer (pybind11 + CUDA)
        CMakeExtension('dpretrieval', 'DPRetrieval'),  # DBoW2 loop-closure retrieval (pybind11)
    ],
    cmdclass={
        'build_ext': BuildExt
    })
