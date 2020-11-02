cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME PolynomialInterpolation)
project(${PROJECT_NAME})

# Set CMake behavior
cmake_policy(SET CMP0004 NEW)

# Executable
add_executable(${PROJECT_NAME} main_copy.cpp)
