#pragma once

#include <cassert>
#include <cmath>
#include <iostream>

namespace nocturne {
namespace geometry {

constexpr float kPi = M_PI;

class Vector2D {
 public:
  Vector2D() = default;
  Vector2D(float x, float y) : x_(x), y_(y) {}

  float x() const { return x_; }
  float y() const { return y_; }

  bool operator==(const Vector2D& v) const { return x_ == v.x_ && y_ == v.y_; }
  bool operator!=(const Vector2D& v) const { return x_ != v.x_ || y_ != v.y_; }

  Vector2D operator-() const { return Vector2D(-x_, -y_); }

  Vector2D operator+(float d) const { return Vector2D(x_ + d, y_ + d); }
  Vector2D& operator+=(float d) {
    x_ += d;
    y_ += d;
    return *this;
  }
  Vector2D operator+(const Vector2D& v) const {
    return Vector2D(x_ + v.x_, y_ + v.y_);
  }
  Vector2D& operator+=(const Vector2D& v) {
    x_ += v.x_;
    y_ += v.y_;
    return *this;
  }

  Vector2D operator-(float d) const { return Vector2D(x_ - d, y_ - d); }
  Vector2D& operator-=(float d) {
    x_ -= d;
    y_ -= d;
    return *this;
  }
  Vector2D operator-(const Vector2D& v) const {
    return Vector2D(x_ - v.x_, y_ - v.y_);
  }
  Vector2D& operator-=(const Vector2D& v) {
    x_ -= v.x_;
    y_ -= v.y_;
    return *this;
  }

  Vector2D operator*(float c) const { return Vector2D(x_ * c, y_ * c); }
  Vector2D& operator*=(float c) {
    x_ *= c;
    y_ *= c;
    return *this;
  }

  Vector2D operator/(float c) const { return Vector2D(x_ / c, y_ / c); }
  Vector2D& operator/=(float c) {
    x_ /= c;
    y_ /= c;
    return *this;
  }

  float Norm(int64_t p = 2) const {
    assert(p > 0);
    switch (p) {
      case 1: {
        return std::fabs(x_) + std::fabs(y_);
      }
      case 2: {
        return std::sqrt(x_ * x_ + y_ * y_);
      }
      case 3: {
        return std::cbrt(std::fabs(x_ * x_ * x_) + std::fabs(y_ * y_ * y_));
      }
      default: {
        const float pp = static_cast<float>(p);
        return std::pow(
            std::pow(std::fabs(x_), pp) + std::pow(std::fabs(y_), pp),
            1.0f / pp);
      }
    }
  }

  float Atan2() const { return std::atan2(y_, x_); }

  Vector2D Rotate(float theta) const {
    const float sin_theta = std::sin(theta);
    const float cos_theta = std::cos(theta);
    return Vector2D(x_ * cos_theta - y_ * sin_theta,
                    x_ * sin_theta + y_ * cos_theta);
  }

 protected:
  float x_ = 0.0f;
  float y_ = 0.0f;
};

Vector2D PolarToVector2D(float r, float theta) {
  return Vector2D(r * std::cos(theta), r * std::sin(theta));
}

float DotProduct(const Vector2D& a, const Vector2D& b) {
  return a.x() * b.x() + a.y() * b.y();
}

float CrossProduct(const Vector2D& a, const Vector2D& b) {
  return a.x() * b.y() - a.y() * b.x();
}

std::ostream& operator<<(std::ostream& os, const Vector2D& v) {
  os << "(" << v.x() << ", " << v.y() << ")";
  return os;
}

}  // namespace geometry
}  // namespace nocturne
