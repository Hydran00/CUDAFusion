cmake_minimum_required(VERSION 3.18)
project(DynamicFusion LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# CUDA settings
set(CMAKE_CUDA_ARCHITECTURES 75 86) # Turing / Ampere — adjust to your GPU
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -lineinfo")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Open3D REQUIRED HINTS /home/hydran00/Open3D/lib/cmake/Open3D)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Library
file(GLOB_RECURSE SOURCES src/*.cpp src/*.cu kernels/*.cu)
add_library(dfusion STATIC ${SOURCES})
set_target_properties(dfusion PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(dfusion
    ${OpenCV_LIBS}
    Eigen3::Eigen
    CUDA::cudart
    CUDA::cublas
)

# Main executable
add_executable(run_fusion tests/main.cu)
set_target_properties(run_fusion PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(run_fusion dfusion Open3D::Open3D)

# Test loader
# add_executable(test_tsdf tests/test_tsdf.cpp)
# target_link_libraries(test_tsdf dfusion)
