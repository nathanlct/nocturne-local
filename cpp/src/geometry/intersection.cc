#include "geometry/intersection.h"

#include <algorithm>
#include <limits>

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
bool Intersects(const AABB& aabb, const LineSegment& segment) {
  const float min_x = aabb.MinX();
  const float min_y = aabb.MinY();
  const float max_x = aabb.MaxX();
  const float max_y = aabb.MaxY();

  float x0 = segment.Endpoint0().x();
  float y0 = segment.Endpoint0().y();
  float x1 = segment.Endpoint1().x();
  float y1 = segment.Endpoint1().y();
  int code0 = ComputeOutCode(aabb, segment.Endpoint0());
  int code1 = ComputeOutCode(aabb, segment.Endpoint1());

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

bool Intersects(const LineSegment& segment, const AABB& aabb) {
  return Intersects(aabb, segment);
}

// Assume the vertices of polygon are in counterclockwise order.
bool Intersects(const ConvexPolygon& polygon, const LineSegment& segment) {
  if (segment.Endpoint0() == segment.Endpoint1()) {
    return polygon.Contains(segment.Endpoint0());
  }

  // Check if polygon lies on the same side of segment.
  const Vector2D d = segment.Endpoint1() - segment.Endpoint0();
  float min_v = std::numeric_limits<float>::max();
  float max_v = std::numeric_limits<float>::lowest();
  for (const Vector2D& p : polygon.vertices()) {
    const float cur = CrossProduct(p - segment.Endpoint0(), d);
    min_v = std::min(min_v, cur);
    max_v = std::max(max_v, cur);
  }
  if ((min_v < 0.0f && max_v < 0.0f) || (min_v > 0.0f && max_v > 0.0f)) {
    return false;
  }

  // Check if segment lies on the same side of one of the edges of polygon.
  const std::vector<LineSegment> edges = polygon.Edges();
  for (const LineSegment& edge : edges) {
    const Vector2D cur_d = edge.Endpoint1() - edge.Endpoint0();
    const float v1 =
        CrossProduct(segment.Endpoint0() - edge.Endpoint0(), cur_d);
    const float v2 =
        CrossProduct(segment.Endpoint1() - edge.Endpoint0(), cur_d);
    if (v1 > 0.0f && v2 > 0.0f) {
      return false;
    }
  }

  return true;
}

bool Intersects(const LineSegment& segment, const ConvexPolygon& polygon) {
  return Intersects(polygon, segment);
}

}  // namespace geometry
}  // namespace nocturne
