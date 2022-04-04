#include "geometry/circular_sector.h"

#include <algorithm>
#include <cassert>

namespace nocturne {
namespace geometry {

namespace {

void CheckMinMaxCoordinates(const Vector2D& p, float& min_x, float& min_y,
                            float& max_x, float& max_y) {
  min_x = std::min(min_x, p.x());
  min_y = std::min(min_y, p.y());
  max_x = std::max(max_x, p.x());
  max_y = std::max(max_y, p.y());
}

}  // namespace

AABB CircularSector::GetAABB() const {
  const Vector2D p0 = center_ + Radius0();
  const Vector2D p1 = center_ + Radius1();
  float min_x = std::min({center_.x(), p0.x(), p1.x()});
  float min_y = std::min({center_.y(), p0.y(), p1.y()});
  float max_x = std::max({center_.x(), p0.x(), p1.x()});
  float max_y = std::max({center_.y(), p0.y(), p1.y()});

  // TODO: Optimize this.
  const Vector2D q0 = center_ + Vector2D(radius_, 0.0f);
  if (CenterAngleContains(q0)) {
    CheckMinMaxCoordinates(q0, min_x, min_y, max_x, max_y);
  }
  const Vector2D q1 = center_ + Vector2D(0.0f, radius_);
  if (CenterAngleContains(q1)) {
    CheckMinMaxCoordinates(q1, min_x, min_y, max_x, max_y);
  }
  const Vector2D q2 = center_ - Vector2D(radius_, 0.0f);
  if (CenterAngleContains(q2)) {
    CheckMinMaxCoordinates(q2, min_x, min_y, max_x, max_y);
  }
  const Vector2D q3 = center_ - Vector2D(0.0f, radius_);
  if (CenterAngleContains(q3)) {
    CheckMinMaxCoordinates(q3, min_x, min_y, max_x, max_y);
  }

  return AABB(min_x, min_y, max_x, max_y);
}

bool CircularSector::Contains(const Vector2D& p) const {
  return Distance(p, center_) <= radius_ && CenterAngleContains(p);
}

bool CircularSector::CenterAngleContains(const Vector2D& p) const {
  const Vector2D d = p - center_;
  const Vector2D r0 = Radius0();
  const Vector2D r1 = Radius1();
  return theta_ < 0.0f
             ? (CrossProduct(d, r0) <= 0.0f || CrossProduct(d, r1) >= 0.0f)
             : (CrossProduct(d, r0) <= 0.0f && CrossProduct(d, r1) >= 0.0f);
}

}  // namespace geometry
}  // namespace nocturne
