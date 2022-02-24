#pragma once

#include "geometry/aabb.h"
#include "geometry/line_segment.h"

namespace nocturne {
namespace geometry {

bool Intersects(const AABB& aabb, const LineSegment& seg);
bool Intersects(const LineSegment& seg, const AABB& aabb);

}  // namespace geometry
}  // namespace nocturne
