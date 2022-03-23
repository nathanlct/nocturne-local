cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(
  CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -Wno-comment -fPIC \
  -march=native -Wfatal-errors -fvisibility=hidden"
)

project(nocturne)

find_package(pybind11 REQUIRED)

add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp
  ${CMAKE_CURRENT_BINARY_DIR}/cpp
)

pybind11_add_module(
  nocturne
  ${CMAKE_CURRENT_SOURCE_DIR}/python/maths.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/python/nocturne.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/python/object.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/python/road.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/python/scenario.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/python/simulation.cpp
)
target_include_directories(
  nocturne
  PUBLIC
  ${CMAKE_SOURCE_DIR}/cpp/include/nocturne_bits
)
target_link_libraries(nocturne PUBLIC nocturne_core)
