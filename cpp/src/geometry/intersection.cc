#include "geometry/intersection.h"

namespace nocturne {
namespace geometry {

// TODO: Find a better way to do this.
bool Intersects(const AABB& aabb, const LineSegment& seg) {
  if (aabb.Contains(seg.Endpoint0()) || aabb.Contains(seg.Endpoint1())) {
    return true;
  }
  const float min_x = aabb.MinX();
  const float min_y = aabb.MinY();
  const float max_x = aabb.MaxX();
  const float max_y = aabb.MaxY();
  return seg.Intersects(
             LineSegment(Vector2D(min_x, min_y), Vector2D(max_x, min_y))) ||
         seg.Intersects(
             LineSegment(Vector2D(max_x, min_y), Vector2D(max_x, max_y))) ||
         seg.Intersects(
             LineSegment(Vector2D(max_x, max_y), Vector2D(min_x, max_y))) ||
         seg.Intersects(
             LineSegment(Vector2D(min_x, max_y), Vector2D(min_x, min_y)));
}

bool Intersects(const LineSegment& seg, const AABB& aabb) {
  return Intersects(aabb, seg);
}

}  // namespace geometry
}  // namespace nocturne
