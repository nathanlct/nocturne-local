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

add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp
  ${CMAKE_CURRENT_BINARY_DIR}/cpp
)

# pybind11
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11
  ${CMAKE_CURRENT_BINARY_DIR}/third_party/pybind11
)

pybind11_add_module(
  nocturne
  ${CMAKE_CURRENT_SOURCE_DIR}/python/nocturne.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/python/object.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/python/road.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/python/scenario.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/python/simulation.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/python/vector_2d.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/python/vehicle.cc
)
target_include_directories(
  nocturne
  PUBLIC
  ${CMAKE_SOURCE_DIR}/cpp/include/nocturne_bits
  ${CMAKE_SOURCE_DIR}/python
)
target_link_libraries(nocturne PUBLIC nocturne_core)
