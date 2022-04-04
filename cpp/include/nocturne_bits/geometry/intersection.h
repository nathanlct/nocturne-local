#pragma once

#include <optional>
#include <utility>

#include "geometry/aabb.h"
#include "geometry/circle.h"
#include "geometry/circular_sector.h"
#include "geometry/line_segment.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

bool Intersects(const AABB& aabb, const LineSegment& segment);
bool Intersects(const LineSegment& segment, const AABB& aabb);

bool Intersects(const ConvexPolygon& polygon, const LineSegment& segment);
bool Intersects(const LineSegment& segment, const ConvexPolygon& polygon);

std::pair<std::optional<Vector2D>, std::optional<Vector2D>> Intersection(
    const Circle& circle, const LineSegment& segment);
std::pair<std::optional<Vector2D>, std::optional<Vector2D>> Intersection(
    const LineSegment& segment, const Circle& circle);

std::pair<std::optional<Vector2D>, std::optional<Vector2D>> Intersection(
    const CircularSector& circular_sector, const LineSegment& segment);
std::pair<std::optional<Vector2D>, std::optional<Vector2D>> Intersection(
    const LineSegment& segment, const CircularSector& circular_sector);

}  // namespace geometry
}  // namespace nocturne
