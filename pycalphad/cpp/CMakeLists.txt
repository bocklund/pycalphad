cmake_minimum_required(VERSION 3.15)
project(pycalphad_cpp VERSION 0.11.1 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
endif()

# Find required packages
find_package(Python3 REQUIRED COMPONENTS Development NumPy)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

# Source files will be added as we migrate modules
set(UTILS_SOURCES
    utils/numpy_wrapper.cpp
    utils/lapack_wrapper.cpp
    utils/memory_utils.cpp
)

set(CORE_SOURCES
    # Will be populated during migration
    # src/hyperplane.cpp
    # src/phase_rec.cpp
    # src/composition_set.cpp
    # src/eqsolver.cpp
    # src/minimizer.cpp
)

# Create static library for C++ core
add_library(pycalphad_core STATIC ${UTILS_SOURCES} ${CORE_SOURCES})
target_link_libraries(pycalphad_core PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
target_include_directories(pycalphad_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Position Independent Code is required for shared libraries
set_target_properties(pycalphad_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Installation rules
install(TARGETS pycalphad_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include/pycalphad
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Optional: Build tests
option(BUILD_TESTS "Build C++ tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
