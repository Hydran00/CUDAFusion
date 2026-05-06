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

message(STATUS "Open3D include dirs: ${Open3D_INCLUDE_DIRS}")
message(STATUS "Open3D libraries: ${Open3D_LIBRARIES}")


set(BUILD_TESTS OFF CACHE BOOL "Build atomic unit tests for individual components")
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
    ${Open3D_INCLUDE_DIRS}
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
    Open3D::Open3D
)

# Main executable
add_executable(run_fusion tests/main.cu)
set_target_properties(run_fusion PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(run_fusion PRIVATE dfusion Open3D::Open3D cudasift_lib)

# ─────────────────────────────────────────────────────────────────
# UNIT TESTS (Atomic component testing)
# Enable with: cmake -DBUILD_TESTS=ON ..
# ─────────────────────────────────────────────────────────────────

if(BUILD_TESTS)
    message(STATUS "Building unit tests (BUILD_TESTS=ON)")
    enable_testing()

    # Test 1: Pose Estimation
    # Tests: rigid transformation estimation, point-to-plane ICP, ground truth validation
    add_executable(test_pose_estimation tests/test_pose_estimation.cu)
    set_target_properties(test_pose_estimation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_pose_estimation
        dfusion cudasift_lib
    )
    add_test(
        NAME PoseEstimation
        COMMAND test_pose_estimation
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    # Test 2: Warp Field
    # Tests: deformation graph, node creation, warping operations
    add_executable(test_warp_field
        tests/test_warp_field.cu
        src/warp_field.cu
    )
    set_target_properties(test_warp_field PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_warp_field
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS}
        Eigen3::Eigen
        CUDA::cudart
        CUDA::cublas
    )
    add_test(
        NAME WarpField
        COMMAND test_warp_field
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    # Test 3: TSDF Integration
    # Tests: volumetric fusion, depth integration, raycasting, volume statistics
    add_executable(test_tsdf_integration
        tests/test_tsdf_integration.cu
        src/tsdf_volume.cu
        kernels/tsdf_kernels.cu
    )
    set_target_properties(test_tsdf_integration PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_tsdf_integration
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS}
        Eigen3::Eigen
        CUDA::cudart
        CUDA::cublas
    )
    add_test(
        NAME TSDFIntegration
        COMMAND test_tsdf_integration
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    # Test 4: Mesh Extraction
    # Tests: marching cubes, surface extraction, mesh quality, PLY export
    add_executable(test_mesh_extraction
        tests/test_mesh_extraction.cu
        src/tsdf_volume.cu
        kernels/tsdf_kernels.cu
    )
    set_target_properties(test_mesh_extraction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_mesh_extraction
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS}
        Open3D::Open3D
        Eigen3::Eigen
        CUDA::cudart
        CUDA::cublas
    )
    add_test(
        NAME MeshExtraction
        COMMAND test_mesh_extraction
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    # Print test information
    message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    message(STATUS "Unit Tests Configuration")
    message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    message(STATUS "✓ test_pose_estimation   - Pose estimation via ICP")
    message(STATUS "✓ test_warp_field       - Deformation graph operations")
    message(STATUS "✓ test_tsdf_integration - Volumetric TSDF fusion")
    message(STATUS "✓ test_mesh_extraction  - Surface mesh extraction")
    message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    message(STATUS "Run tests with: ctest --output-on-failure")
    message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

else()
    message(STATUS "Unit tests disabled (BUILD_TESTS=OFF)")
    message(STATUS "Enable with: cmake -DBUILD_TESTS=ON ..")
endif()

