#pragma once

#include <cmath>

namespace nocturne {
namespace geometry {
namespace utils {

constexpr double kEps = 1e-8;
constexpr double kPi = M_PI;

template <typename T>
inline bool AlmostEquals(const T& lhs, const T& rhs, const T& eps = kEps) {
  return std::fabs(lhs - rhs) < eps;
}

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
