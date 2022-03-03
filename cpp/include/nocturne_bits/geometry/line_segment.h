#pragma once

#include <algorithm>
#include <array>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/geometry_utils.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

class LineSegment : public AABBInterface {
 public:
  LineSegment() = default;
  LineSegment(const Vector2D& p, const Vector2D& q) {
    endpoints_[0] = p;
    endpoints_[1] = q;
  }
  LineSegment(const LineSegment& seg) {
    endpoints_[0] = seg.endpoints_[0];
    endpoints_[1] = seg.endpoints_[1];
  }

  const Vector2D& Endpoint0() const { return endpoints_[0]; }
  const Vector2D& Endpoint1() const { return endpoints_[1]; }
  const Vector2D& Endpoint(int64_t index) const { return endpoints_.at(index); }

  float Length() const { return Distance(endpoints_[0], endpoints_[1]); }

  AABB GetAABB() const override {
    const float min_x = std::min(endpoints_[0].x(), endpoints_[1].x());
    const float max_x = std::max(endpoints_[0].x(), endpoints_[1].x());
    const float min_y = std::min(endpoints_[0].y(), endpoints_[1].y());
    const float max_y = std::max(endpoints_[0].y(), endpoints_[1].y());
    return AABB(min_x, min_y, max_x, max_y);
  }

  Vector2D NormalVector() const {
    const Vector2D d = endpoints_[1] - endpoints_[0];
    return Vector2D(-d.y(), d.x()) / d.Norm();
  }

  bool Contains(const Vector2D& o) const {
    return DotProduct(endpoints_[0] - o, endpoints_[1] - o) <= 0;
  }

  bool Intersects(const LineSegment& seg) const;

 protected:
  std::array<Vector2D, 2> endpoints_;
};

}  // namespace geometry
}  // namespace nocturne
