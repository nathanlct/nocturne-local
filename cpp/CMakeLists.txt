cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(nocturne VERSION 0.1.0)

# Include dir
include_directories(/usr/local/include)

# Src
AUX_SOURCE_DIRECTORY(src SRC_FILES)

# Headers
set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR "include/nocturne_bits")

# Source files
file (GLOB SOURCE_FILES "${PROJECT_INCLUDE_DIR}/*.hpp" "${PROJECT_SOURCE_DIR}/*.cpp")

# Set up such that XCode organizes the files correctly
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})

# Add library
add_library(nocturne SHARED ${SOURCE_FILES})

# Include directories
target_include_directories(nocturne PRIVATE include/nocturne_bits)
find_package(SFML 2.5 COMPONENTS window system graphics audio REQUIRED)
target_link_libraries(nocturne PUBLIC sfml-window sfml-system sfml-graphics sfml-audio)

# Install
install(TARGETS nocturne DESTINATION lib)

# Install the headers
install(FILES include/nocturne DESTINATION include)

# Create base directory
install(DIRECTORY include/nocturne_bits DESTINATION include)