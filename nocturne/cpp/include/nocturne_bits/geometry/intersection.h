#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/circle.h"
#include "geometry/circular_sector.h"
#include "geometry/line_segment.h"
#include "geometry/point_like.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

inline bool Intersects(const AABB& lhs, const AABB& rhs) {
  return lhs.Intersects(rhs);
}

bool Intersects(const AABB& aabb, const LineSegment& segment);
bool Intersects(const LineSegment& segment, const AABB& aabb);

bool Intersects(const ConvexPolygon& polygon, const LineSegment& segment);
bool Intersects(const LineSegment& segment, const ConvexPolygon& polygon);

std::vector<int32_t> BatchIntersects(
    const ConvexPolygon& polygon, const Vector2D& o,
    const std::vector<const PointLike*>& points);

inline bool Intersects(const ConvexPolygon& lhs, const ConvexPolygon& rhs) {
  return lhs.Intersects(rhs);
}

inline std::pair<std::optional<Vector2D>, std::optional<Vector2D>> Intersection(
    const Circle& circle, const LineSegment& segment) {
  return circle.Intersection(segment);
}

inline std::pair<std::optional<Vector2D>, std::optional<Vector2D>> Intersection(
    const LineSegment& segment, const Circle& circle) {
  return circle.Intersection(segment);
}

inline std::pair<std::optional<Vector2D>, std::optional<Vector2D>> Intersection(
    const CircularSector& circular_sector, const LineSegment& segment) {
  return circular_sector.Intersection(segment);
}

inline std::pair<std::optional<Vector2D>, std::optional<Vector2D>> Intersection(
    const LineSegment& segment, const CircularSector& circular_sector) {
  return circular_sector.Intersection(segment);
}

}  // namespace geometry
}  // namespace nocturne
