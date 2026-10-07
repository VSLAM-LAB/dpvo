cmake_minimum_required(VERSION 3.4...3.18)
project(dpretrieval)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# DBoW2 (git submodule ../DBoW2) is compiled straight into the module instead of being installed as a library
set(DBOW2_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../DBoW2)
set(DBOW2_SRCS  # same list as DBoW2's own CMakeLists (FSurf64 needs OpenCV's non-free SURF)
  ${DBOW2_DIR}/src/BowVector.cpp     ${DBOW2_DIR}/src/FBrief.cpp        ${DBOW2_DIR}/src/FORB.cpp
  ${DBOW2_DIR}/src/FeatureVector.cpp ${DBOW2_DIR}/src/QueryResults.cpp  ${DBOW2_DIR}/src/ScoringObject.cpp)
include_directories(${DBOW2_DIR}/include ${DBOW2_DIR}/include/DBoW2)

find_package(pybind11 REQUIRED)
pybind11_add_module(dpretrieval src/main.cpp ${DBOW2_SRCS})
target_link_libraries(dpretrieval PRIVATE ${OpenCV_LIBS})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(dpretrieval
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
