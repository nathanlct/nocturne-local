#pragma once

#include <pybind11/numpy.h>

#include <cstring>
#include <vector>

namespace py = pybind11;

namespace nocturne {
namespace utils {

template <typename T>
py::array_t<T> ToNumpyArray(const std::vector<T>& vec) {
  py::array_t<T> arr(vec.size());
  std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(T));
  return arr;
}

}  // namespace utils
}  // namespace nocturne
