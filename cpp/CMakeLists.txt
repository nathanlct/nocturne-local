cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(nocturne VERSION 0.1.0)

# nocturne files
include_directories(/usr/local/include)
AUX_SOURCE_DIRECTORY(src SRC_FILES)
set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR "include/nocturne_bits")
file (GLOB SOURCE_FILES "${PROJECT_INCLUDE_DIR}/*.hpp" "${PROJECT_SOURCE_DIR}/*.cpp")

# Build and link nocturne
add_library(nocturne SHARED ${SOURCE_FILES})
target_include_directories(nocturne PRIVATE include/nocturne_bits)

# Link SFML
find_package(SFML 2.5 COMPONENTS window system graphics audio REQUIRED)
target_link_libraries(nocturne PUBLIC sfml-window sfml-system sfml-graphics sfml-audio)

# Install library and includes
install(TARGETS nocturne DESTINATION lib)
install(FILES include/nocturne DESTINATION include)
install(DIRECTORY include/nocturne_bits DESTINATION include)