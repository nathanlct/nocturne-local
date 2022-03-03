cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(nocturne)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(
  CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -Wno-comment -fPIC \
  -march=native -Wfatal-errors -fvisibility=hidden"
)

# project(test_simulation)
#
# # Set output dirs
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin)
#
# # Find SFML and nocturne
# find_package(SFML 2.5 COMPONENTS window system graphics audio REQUIRED)
# find_library(NOCTURNE_LIB nocturne HINTS /usr/local/lib/)
# include_directories(/usr/local/include ${ADDITIONAL_INCLUDE_DIR})
#
# # Create binaries
# add_executable(test_simulation src/test_simulation.cpp)
#
# # Link SFML and nocturne
# target_link_libraries(test_simulation PUBLIC sfml-window sfml-system sfml-graphics sfml-audio)
# target_link_libraries(test_simulation PUBLIC ${NOCTURNE_LIB})

add_executable(
  nocturne_test 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/bvh_test.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/intersection_test.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/line_segment_test.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/polygon_test.cc
)
target_include_directories(
  nocturne_test
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/nocturne_bits
)
target_link_libraries(
  nocturne_test
  PUBLIC
  gmock
  gtest
  gtest_main
  nocturne_core
)

include(GoogleTest)
gtest_discover_tests(nocturne_test)
