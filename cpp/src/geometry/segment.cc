#include "geometry/segment.h"

namespace nocturne {
namespace geometry {

namespace {

bool CCW(const Vector2D& x, const Vector2D& p, const Vector2D& q) {
  return CrossProduct(x - p, q - p) < 0.0f;
}

}  // namespace

bool Segment::Intersects(const Segment& seg) const {
  const Vector2D& p1 = endpoints_[0];
  const Vector2D& q1 = endpoints_[1];
  const Vector2D& p2 = seg.endpoints_[0];
  const Vector2D& q2 = seg.endpoints_[1];
  return CCW(p1, p2, q2) != CCW(q1, p2, q2) &&
         CCW(p2, p1, q1) != CCW(q2, p1, q1);
}

}  // namespace geometry
}  // namespace nocturne
