#pragma once

#include <cmath>

namespace nocturne {
namespace geometry {
namespace utils {

constexpr double kPi = M_PI;

template <typename T>
constexpr T Radians(const T& d) {
  return d / T(180.0) * static_cast<T>(kPi);
}

template <typename T>
constexpr T Degrees(const T& r) {
  return r / static_cast<T>(kPi) * T(180.0);
}

}  // namespace utils
}  // namespace geometry
}  // namespace nocturne
