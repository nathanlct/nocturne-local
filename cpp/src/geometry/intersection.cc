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

std::optional<Vector2D> ArcLineIntersection(
    const CircularSector& circular_sector, const LineSegment& segment,
    float t) {
  if (t >= 0.0f && t <= 1.0f) {
    const Vector2D x = segment.Point(t);
    if (circular_sector.Contains(x)) {
      return x;
    }
  }
  return std::nullopt;
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

std::pair<std::optional<Vector2D>, std::optional<Vector2D>> Intersection(
    const CircularSector& circular_sector, const LineSegment& segment) {
  std::array<Vector2D, 2> ret;
  int64_t cnt = 0;

  const Vector2D& o = circular_sector.center();
  const LineSegment edge0(o, o + circular_sector.Radius0());
  const LineSegment edge1(o, o + circular_sector.Radius1());
  const auto u = edge0.Intersection(segment);
  if (u.has_value()) {
    ret[cnt++] = *u;
  }
  const auto v = edge1.Intersection(segment);
  if (v.has_value()) {
    ret[cnt++] = *v;
  }
  if (cnt == 2) {
    return std::make_pair<std::optional<Vector2D>, std::optional<Vector2D>>(
        std::make_optional(ret[0]), std::make_optional(ret[1]));
  }

  const Vector2D& p = segment.Endpoint0();
  const Vector2D& q = segment.Endpoint1();
  const Vector2D d1 = q - p;
  const Vector2D d2 = p - o;
  const float r = circular_sector.radius();
  const float a = DotProduct(d1, d1);
  const float b = DotProduct(d1, d2) * 2.0f;
  const float c = DotProduct(d2, d2) - r * r;
  const float delta = b * b - 4.0f * a * c;
  if (utils::AlmostEquals(delta, 0.0f)) {
    const float t = -b / (2.0f * a);
    const auto x = ArcLineIntersection(circular_sector, segment, t);
    if (x.has_value()) {
      ret[cnt++] = *x;
    }
  } else if (delta > 0.0f) {
    const float t0 = (-b - std::sqrt(delta)) / (2.0f * a);
    const float t1 = (-b + std::sqrt(delta)) / (2.0f * a);
    const auto x = ArcLineIntersection(circular_sector, segment, t0);
    const auto y = ArcLineIntersection(circular_sector, segment, t1);
    if (x.has_value()) {
      ret[cnt++] = *x;
    }
    if (y.has_value()) {
      ret[cnt++] = *y;
    }
  }

  if (cnt == 0) {
    return std::make_pair<std::optional<Vector2D>, std::optional<Vector2D>>(
        std::nullopt, std::nullopt);
  } else if (cnt == 1) {
    return std::make_pair<std::optional<Vector2D>, std::optional<Vector2D>>(
        std::make_optional(ret[0]), std::nullopt);
  } else {
    return std::make_pair<std::optional<Vector2D>, std::optional<Vector2D>>(
        std::make_optional(ret[0]), std::make_optional(ret[1]));
  }
}

std::pair<std::optional<Vector2D>, std::optional<Vector2D>> Intersection(
    const LineSegment& segment, const CircularSector& circular_sector) {
  return Intersection(circular_sector, segment);
}

}  // namespace geometry
}  // namespace nocturne
