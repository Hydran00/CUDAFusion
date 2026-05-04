cmake_minimum_required(VERSION 3.18)
project(DynamicFusion LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# CUDA settings
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math --expt-relaxed-constexpr -Xptxas=-O3")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Open3D REQUIRED HINTS /home/hydran00/Open3D/lib/cmake/Open3D)
include(FetchContent)

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(VERBOSE_OUTPUT OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/CudaSift)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.9.0 # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_MakeAvailable(yaml-cpp)


include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty/CudaSift
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
    yaml-cpp::yaml-cpp
)

# Main executable
add_executable(run_fusion tests/main.cu)
set_target_properties(run_fusion PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(run_fusion dfusion Open3D::Open3D)
target_link_libraries(run_fusion cudasift_lib)

# Test loader
# add_executable(test_tsdf tests/test_tsdf.cpp)
# target_link_libraries(test_tsdf dfusion)
