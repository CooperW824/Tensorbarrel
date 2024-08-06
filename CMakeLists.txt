cmake_minimum_required(VERSION 3.0)

project(Tensorbarrel)

# Set the C++ standard to C++20

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the version number
set(Tensorbarrel VERSION 0.1.0)

# Update the version number in the header file
configure_file(
    "src/version.h.in"
    "include/Tensorbarrel/version.h"
)

# Set the build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set the output directory for the build
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set the output directory for the libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(Tensorbarrel INTERFACE)

target_include_directories(Tensorbarrel INTERFACE include)
