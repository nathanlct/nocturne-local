#pragma once

#include <cmath>

namespace nocturne {
namespace geometry {
namespace utils {

constexpr double kEps = 1e-8;
constexpr double kPi = M_PI;
constexpr double kTau = M_2_PI;
constexpr double kTwoPi = M_2_PI;
constexpr double kHalfPi = M_PI_2;

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

inline float AngleAdd(float lhs, float rhs) {
  const float ret = std::fmod(lhs + rhs, static_cast<float>(kTau));
  return ret < 0.0f ? ret + static_cast<float>(kTau) : ret;
}

inline double AngleAdd(double lhs, double rhs) {
  const double ret = std::fmod(lhs + rhs, kTau);
  return ret < 0.0 ? ret + kTau : ret;
}

inline float AngleSub(float lhs, float rhs) {
  const float ret = std::fmod(lhs - rhs, static_cast<float>(kTau));
  return ret < 0.0f ? ret + static_cast<float>(kTau) : ret;
}

inline double AngleSub(double lhs, double rhs) {
  const double ret = std::fmod(lhs - rhs, kTau);
  return ret < 0.0 ? ret + kTau : ret;
}

}  // namespace utils
}  // namespace geometry
}  // namespace nocturne
