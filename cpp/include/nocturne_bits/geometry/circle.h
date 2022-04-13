#pragma once

#include <optional>
#include <utility>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/line_segment.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

class CircleLike : public AABBInterface {
 public:
  CircleLike() = default;
  CircleLike(const Vector2D& center, float radius)
      : center_(center), radius_(radius) {}

  const Vector2D& center() const { return center_; }
  float radius() const { return radius_; }

  virtual float Area() const = 0;
  virtual bool Contains(const Vector2D& p) const = 0;
  virtual std::pair<std::optional<Vector2D>, std::optional<Vector2D>>
  Intersection(const LineSegment& segment) const;

 protected:
  const Vector2D center_;
  const float radius_;
};

class Circle : public CircleLike {
 public:
  Circle() = default;
  Circle(const Vector2D& center, float radius) : CircleLike(center, radius) {}

  AABB GetAABB() const override {
    return AABB(center_ - radius_, center_ + radius_);
  }

  float Area() const override { return utils::kPi * radius_ * radius_; }

  bool Contains(const Vector2D& p) const override {
    return Distance(center_, p) <= radius_;
  }
};

}  // namespace geometry
}  // namespace nocturne
