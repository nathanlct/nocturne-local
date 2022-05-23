#pragma once

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace nocturne {
namespace geometry {

class Vector2D;
class PointLike;

namespace utils {

constexpr double kEps = 1e-8;
constexpr double kPi = M_PI;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kHalfPi = M_PI_2;
constexpr double kQuarterPi = M_PI_4;

template <typename T>
inline bool AlmostEquals(T lhs, T rhs) {
  return std::fabs(lhs - rhs) < std::numeric_limits<T>::epsilon() * T(32);
}

template <typename T>
constexpr T Radians(T d) {
  return d / 180.0 * kPi;
}

template <typename T>
constexpr T Degrees(T r) {
  return r / kPi * 180.0;
}

// Check if angle is in the range of [-Pi, Pi].
template <typename T>
constexpr bool IsNormalizedAngle(T angle) {
  return angle >= -kPi && angle <= kPi;
}

template <typename T>
inline float NormalizeAngle(T angle) {
  const T ret = std::fmod(angle, kTwoPi);
  return ret > kPi ? ret - kTwoPi : (ret < -kPi ? ret + kTwoPi : ret);
}

template <typename T>
inline T AngleAdd(T lhs, T rhs) {
  return NormalizeAngle<T>(lhs + rhs);
}

template <typename T>
inline T AngleSub(T lhs, T rhs) {
  return NormalizeAngle<T>(lhs - rhs);
}

std::pair<std::vector<float>, std::vector<float>> PackCoordinates(
    const std::vector<Vector2D>& points);
std::pair<std::vector<float>, std::vector<float>> PackCoordinates(
    const std::vector<const PointLike*>& points);

}  // namespace utils
}  // namespace geometry
}  // namespace nocturne
