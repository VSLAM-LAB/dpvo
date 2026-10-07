cmake_minimum_required(VERSION 3.8)

project(dpviewer)
project(viewer LANGUAGES CUDA CXX)

# The C++ ABI comes from Torch's CMake config, and CMAKE_CUDA_ARCHITECTURES from setup.py (TORCH_CUDA_ARCH_LIST,
# or "native" for the local GPU)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS}")


list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})

find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)
# Torch's CMake config links CUDA::nvToolsExt, which CUDA 12 no longer ships (NVTX3 is header-only)
if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

include_directories(
    ${EIGEN_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS} 
) 

add_subdirectory(dpviewer)


