#pragma once

#include "geometry/aabb.h"
#include "geometry/line_segment.h"
#include "geometry/polygon.h"

namespace nocturne {
namespace geometry {

bool Intersects(const AABB& aabb, const LineSegment& segment);
bool Intersects(const LineSegment& segment, const AABB& aabb);

bool Intersects(const ConvexPolygon& polygon, const LineSegment& segment);
bool Intersects(const LineSegment& segment, const ConvexPolygon& polygon);

}  // namespace geometry
}  // namespace nocturne
