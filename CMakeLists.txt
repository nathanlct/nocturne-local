cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# fix for bug (https://gitlab.kitware.com/cmake/cmake/-/issues/17995)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(nocturne)

# Find project files
include_directories("${CMAKE_SOURCE_DIR}/cpp/include/nocturne_bits")
include_directories("${CMAKE_SOURCE_DIR}/python")

file (GLOB SOURCE_FILES "cpp/src/*.cpp")
file (GLOB HEADER_FILES "cpp/include/nocturne_bits/*.hpp")
file (GLOB PYTHON_FILES "python/*.cpp" "python/*.hpp")

# Find pybind and SFML
find_package(pybind11 REQUIRED)
find_package(SFML 2.5 COMPONENTS window system graphics audio REQUIRED)

# Create Python module for nocturne
pybind11_add_module(nocturne 
	${SOURCE_FILES}
	${HEADER_FILES}
	${PYTHON_FILES}
)

target_include_directories(
  nocturne
  PUBLIC
  ${CMAKE_SOURCE_DIR}/cpp/include/nocturne_bits
)

# Link nocturne and SFML
target_link_libraries(nocturne PUBLIC)
target_link_libraries(nocturne PUBLIC sfml-window sfml-system sfml-graphics sfml-audio)

# Install Python module for nocturne
install(TARGETS nocturne
  COMPONENT python
  LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
)
