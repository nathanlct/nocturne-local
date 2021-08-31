cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(test_simulation)

# Set output dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin)

# Find SFML and nocturne
find_package(SFML 2.5 COMPONENTS window system graphics audio REQUIRED)
find_library(NOCTURNE_LIB nocturne HINTS /usr/local/lib/)
include_directories(/usr/local/include ${ADDITIONAL_INCLUDE_DIR})

# Create binaries
add_executable(test_simulation src/test_simulation.cpp)

# Link SFML and nocturne
target_link_libraries(test_simulation PUBLIC sfml-window sfml-system sfml-graphics sfml-audio)
target_link_libraries(test_simulation PUBLIC ${NOCTURNE_LIB})