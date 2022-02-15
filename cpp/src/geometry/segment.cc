#include "geometry/segment.h"

namespace nocturne {
namespace geometry {

namespace {

bool CCW(const Vector2D& a, const Vector2D& b, const Vector2D& c) {
  return CrossProduct(b - a, c - a) > 0.0f;
}

}  // namespace

bool Segment::Intersects(const Segment& seg) const {
  const Vector2D& p1 = endpoints_[0];
  const Vector2D& q1 = endpoints_[1];
  const Vector2D& p2 = seg.endpoints_[0];
  const Vector2D& q2 = seg.endpoints_[1];
  return CCW(p1, q1, p2) != CCW(p1, q1, q2) &&
         CCW(p2, q2, p1) != CCW(p2, q2, q1);
}

}  // namespace geometry
}  // namespace nocturne
