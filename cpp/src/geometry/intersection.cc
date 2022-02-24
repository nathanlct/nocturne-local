#include "geometry/intersection.h"

namespace nocturne {
namespace geometry {

namespace {

constexpr int kInside = 0;
constexpr int kLeft = 1;
constexpr int kRight = 2;
constexpr int kBottom = 4;
constexpr int kTop = 8;

int ComputeOutCode(const AABB& aabb, const Vector2D& p) {
  int code = kInside;
  if (p.x() < aabb.MinX()) {
    code |= kLeft;
  } else if (p.x() > aabb.MaxX()) {
    code |= kRight;
  }
  if (p.y() < aabb.MinY()) {
    code |= kBottom;
  } else if (p.y() > aabb.MaxY()) {
    code |= kTop;
  }
  return code;
}

}  // namespace

// Cohen–Sutherland algorithm
// https://en.wikipedia.org/wiki/Cohen%E2%80%93Sutherland_algorithm
bool Intersects(const AABB& aabb, const LineSegment& seg) {
  const float min_x = aabb.MinX();
  const float min_y = aabb.MinY();
  const float max_x = aabb.MaxX();
  const float max_y = aabb.MaxY();

  float x0 = seg.Endpoint0().x();
  float y0 = seg.Endpoint0().y();
  float x1 = seg.Endpoint1().x();
  float y1 = seg.Endpoint1().y();
  int code0 = ComputeOutCode(aabb, seg.Endpoint0());
  int code1 = ComputeOutCode(aabb, seg.Endpoint1());

  while (true) {
    if ((code0 | code1) == 0) {
      return true;
    }
    if ((code0 & code1) != 0) {
      return false;
    }
    const int code = std::max(code0, code1);
    float x = 0;
    float y = 0;
    if ((code & kTop) != 0) {
      x = x0 + (x1 - x0) * (max_y - y0) / (y1 - y0);
      y = max_y;
    } else if ((code & kBottom) != 0) {
      x = x0 + (x1 - x0) * (min_y - y0) / (y1 - y0);
      y = min_y;
    } else if ((code & kRight) != 0) {
      y = y0 + (y1 - y0) * (max_x - x0) / (x1 - x0);
      x = max_x;
    } else if ((code & kLeft) != 0) {
      y = y0 + (y1 - y0) * (min_x - x0) / (x1 - x0);
      x = min_x;
    }
    if (code == code0) {
      x0 = x;
      y0 = y;
      code0 = ComputeOutCode(aabb, Vector2D(x0, y0));
    } else {
      x1 = x;
      y1 = y;
      code1 = ComputeOutCode(aabb, Vector2D(x1, y1));
    }
  }
  return false;
}

bool Intersects(const LineSegment& seg, const AABB& aabb) {
  return Intersects(aabb, seg);
}

}  // namespace geometry
}  // namespace nocturne
