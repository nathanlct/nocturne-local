#include "geometry/intersection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "geometry/geometry_utils.h"

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

  // Check if segment lies on the right of one of the edges of polygon.
  const std::vector<LineSegment> edges = polygon.Edges();
  for (const LineSegment& edge : edges) {
    const Vector2D cur_d = edge.Endpoint1() - edge.Endpoint0();
    const float v0 =
        CrossProduct(segment.Endpoint0() - edge.Endpoint0(), cur_d);
    const float v1 =
        CrossProduct(segment.Endpoint1() - edge.Endpoint0(), cur_d);
    if (v0 > 0.0f && v1 > 0.0f) {
      return false;
    }
  }

  return true;
}

bool Intersects(const LineSegment& segment, const ConvexPolygon& polygon) {
  return Intersects(polygon, segment);
}

std::vector<int32_t> BatchIntersects(
    const ConvexPolygon& polygon, const Vector2D& o,
    const std::vector<const PointLike*>& points) {
  const int64_t n = points.size();
  std::vector<float> x(n);
  std::vector<float> y(n);
  for (int64_t i = 0; i < n; ++i) {
    const Vector2D p = points[i]->Coordinate();
    x[i] = p.x();
    y[i] = p.y();
  }

  std::vector<int32_t> mask(n, 1);
  std::vector<float> min_v(n, std::numeric_limits<float>::max());
  std::vector<float> max_v(n, std::numeric_limits<float>::lowest());
  const float ox = o.x();
  const float oy = o.y();

  for (const Vector2D& v : polygon.vertices()) {
    const float vx = v.x() - ox;
    const float vy = v.y() - oy;
    for (int64_t i = 0; i < n; ++i) {
      const float dx = x[i] - ox;
      const float dy = y[i] - oy;
      const float cur = vx * dy - dx * vy;
      // std::min and std::max are slow, use conditional operator here.
      min_v[i] = min_v[i] < cur ? min_v[i] : cur;
      max_v[i] = max_v[i] > cur ? max_v[i] : cur;
    }
  }
  for (int64_t i = 0; i < n; ++i) {
    mask[i] &= ((((min_v[i] < 0.0f) & (max_v[i] < 0.0f)) |
                 ((min_v[i] > 0.0f) & (max_v[i] > 0.0f))) ^
                1);
  }

  const std::vector<LineSegment> edges = polygon.Edges();
  for (const LineSegment& edge : edges) {
    const float p0x = edge.Endpoint0().x();
    const float p0y = edge.Endpoint0().y();
    const float p1x = edge.Endpoint1().x();
    const float p1y = edge.Endpoint1().y();
    const float dx = p1x - p0x;
    const float dy = p1y - p0y;
    const float v0x = ox - p0x;
    const float v0y = oy - p0y;
    for (int64_t i = 0; i < n; ++i) {
      const float v1x = x[i] - p0x;
      const float v1y = y[i] - p0y;
      const float v0 = v0x * dy - dx * v0y;
      const float v1 = v1x * dy - dx * v1y;
      mask[i] &= (((v0 > 0.0f) & (v1 > 0.0f)) ^ 1);
    }
  }

  return mask;
}

}  // namespace geometry
}  // namespace nocturne
