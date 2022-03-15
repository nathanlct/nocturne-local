#pragma once

#include <cmath>

namespace nocturne {
namespace geometry {
namespace utils {

constexpr double kEps = 1e-8;
constexpr double kPi = M_PI;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kHalfPi = M_PI_2;
constexpr double kQuarterPi = M_PI_4;

inline bool AlmostEquals(float lhs, float rhs, float eps = kEps) {
  return std::fabs(lhs - rhs) < eps;
}

inline bool AlmostEquals(double lhs, double rhs, double eps = kEps) {
  return std::fabs(lhs - rhs) < eps;
}

constexpr float Radians(float d) {
  return d / 180.0f * static_cast<float>(kPi);
}

constexpr double Radians(double d) { return d / 180.0 * kPi; }

constexpr float Degrees(float r) {
  return r / static_cast<float>(kPi) * 180.0f;
}

constexpr double Degrees(double r) { return r / kPi * 180.0; }

// Check if angle is in the range of [-Pi, Pi].
constexpr bool IsNormalizedAngle(float angle) {
  return angle >= -static_cast<float>(kPi) && angle <= static_cast<float>(kPi);
}

constexpr bool IsNormalizedAngle(double angle) {
  return angle >= -kPi && angle <= kPi;
}

inline float NormalizeAngle(float angle) {
  constexpr float kPiF = kPi;
  constexpr float kTwoPiF = kTwoPi;
  const float ret = std::fmod(angle, kTwoPiF);
  return ret > kPiF ? ret - kTwoPiF : (ret < -kPiF ? ret + kTwoPiF : ret);
}

inline double NormalizeAngle(double angle) {
  const double ret = std::fmod(angle, kTwoPi);
  return ret > kPi ? ret - kTwoPi : (ret < -kPi ? ret + kTwoPi : ret);
}

inline float AngleAdd(float lhs, float rhs) {
  return NormalizeAngle(lhs + rhs);
}

inline double AngleAdd(double lhs, double rhs) {
  return NormalizeAngle(lhs + rhs);
}

inline float AngleSub(float lhs, float rhs) {
  return NormalizeAngle(lhs - rhs);
}

inline double AngleSub(double lhs, double rhs) {
  return NormalizeAngle(lhs - rhs);
}

}  // namespace utils
}  // namespace geometry
}  // namespace nocturne
