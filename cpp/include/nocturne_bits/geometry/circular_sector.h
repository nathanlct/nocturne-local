#pragma once

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/geometry_utils.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

class CircularSector : public AABBInterface {
 public:
  CircularSector() = default;
  CircularSector(const Vector2D& center, float radius, float heading,
                 float theta)
      : center_(center),
        radius_(radius),
        heading_(utils::NormalizeAngle<float>(heading)),
        theta_(utils::NormalizeAngle<float>(theta)) {}

  const Vector2D& center() const { return center_; }
  float radius() const { return radius_; }
  float heading() const { return heading_; }
  float theta() const { return theta_; }

  float Angle0() const {
    return theta_ < 0.0f
               ? utils::AngleSub<float>(heading_, theta_ * 0.5f + utils::kPi)
               : utils::AngleSub<float>(heading_, theta_ * 0.5f);
  }
  float Angle1() const {
    return theta_ < 0.0f
               ? utils::AngleAdd<float>(heading_, theta_ * 0.5f + utils::kPi)
               : utils::AngleAdd<float>(heading_, theta_ * 0.5f);
  }

  Vector2D Radius0() const { return PolarToVector2D(radius_, Angle0()); }
  Vector2D Radius1() const { return PolarToVector2D(radius_, Angle1()); }

  AABB GetAABB() const override;

  float Area() const {
    const float theta = theta_ < 0.0f ? theta_ + utils::kTwoPi : theta_;
    return theta * radius_ * radius_ * 0.5f;
  }

  bool Contains(const Vector2D& p) const;

 protected:
  const Vector2D center_;
  const float radius_;
  const float heading_;
  const float theta_;
};

}  // namespace geometry
}  // namespace nocturne
