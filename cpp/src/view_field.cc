#include "view_field.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

#include "geometry/intersection.h"
#include "geometry/line_segment.h"
#include "geometry/polygon.h"

namespace nocturne {

namespace {

using geometry::CircularSector;
using geometry::ConvexPolygon;
using geometry::LineSegment;
using geometry::Vector2D;

void VisibleObjectsImpl(const LineSegment& sight,
                        const std::vector<const Object*>& objects,
                        std::vector<bool>& mask) {
  const int64_t n = objects.size();

  // Compute relative distance from center to each object.
  std::vector<float> dis(n, std::numeric_limits<float>::max());
  float min_dis = std::numeric_limits<float>::max();
  int64_t min_idx = -1;
  for (int64_t i = 0; i < n; ++i) {
    const Object* obj = objects[i];
    const auto edges = obj->BoundingPolygon().Edges();
    for (const LineSegment& edge : edges) {
      const auto t = sight.ParametricIntersection(edge);
      if (!t.has_value()) {
        continue;
      }
      dis[i] = std::min(dis[i], *t);
      if (obj->can_block_sight() && *t < min_dis) {
        min_dis = *t;
        min_idx = i;
      }
    }
  }

  // Blocking object is visible.
  if (min_idx >= 0) {
    mask[min_idx] = true;
  }

  const Vector2D& o = sight.Endpoint0();
  const Vector2D p = min_idx < 0 ? sight.Endpoint1() : sight.Point(min_dis);
  const LineSegment seg(o, p);
  for (int64_t i = 0; i < n; ++i) {
    // Already visible.
    if (mask[i]) {
      continue;
    }
    const Object* obj = objects[i];
    // Non blocking nearby objects are visible.
    if (!obj->can_block_sight() && dis[i] < min_dis) {
      mask[i] = true;
      continue;
    }
    const ConvexPolygon polygon = obj->BoundingPolygon();
    const auto& vertices = polygon.vertices();
    for (const Vector2D& p : vertices) {
      // Corners of objects are visible.
      if (seg.Contains(p)) {
        mask[i] = true;
        break;
      }
    }
  }
}

}  // namespace

// O(N^2) algorithm.
// TODO: Implment O(NlogN) algorithm when there are too many objects.
std::vector<const Object*> ViewField::VisibleObjects(
    const std::vector<const Object*>& objects, int64_t limit) const {
  const int64_t n = objects.size();
  const Vector2D& o = center();
  const std::vector<Vector2D> sight_endpoints = ComputeSightEndpoints(objects);
  std::vector<bool> mask(n, false);
  for (const Vector2D& p : sight_endpoints) {
    VisibleObjectsImpl(LineSegment(o, p), objects, mask);
  }
  std::vector<const Object*> ret;
  for (int64_t i = 0; i < n; ++i) {
    if (mask[i]) {
      ret.push_back(objects[i]);
    }
  }
  const int64_t m = ret.size();
  return (limit < 0 || m <= limit) ? ret : NearestK(ret, limit);
}

std::vector<const Object*> ViewField::VisibleNonblockingObjects(
    const std::vector<const Object*>& objects, int64_t limit) const {
  std::vector<const Object*> ret;
  for (const Object* obj : objects) {
    const auto edges = obj->BoundingPolygon().Edges();
    for (const LineSegment& edge : edges) {
      // Check one endpoint should be enough, the othe one will be checked in
      // the next edge.
      const Vector2D& x = edge.Endpoint0();
      if (Contains(x)) {
        ret.push_back(obj);
        break;
      }
      const auto [p, q] = Intersection(*this, edge);
      if (p.has_value() || q.has_value()) {
        ret.push_back(obj);
        break;
      }
    }
  }
  const int64_t m = ret.size();
  return (limit < 0 || m <= limit) ? ret : NearestK(ret, limit);
}

std::vector<Vector2D> ViewField::ComputeSightEndpoints(
    const std::vector<const Object*>& objects) const {
  std::vector<Vector2D> ret;
  const Vector2D& o = center();
  ret.push_back(o + Radius0());
  ret.push_back(o + Radius1());
  for (const Object* obj : objects) {
    const auto edges = obj->BoundingPolygon().Edges();
    for (const LineSegment& edge : edges) {
      // Check one endpoint should be enough, the othe one will be checked in
      // the next edge.
      const Vector2D& x = edge.Endpoint0();
      if (Contains(x)) {
        ret.push_back(MakeSightEndpoint(x));
      }
      const auto [p, q] = Intersection(*this, edge);
      if (p.has_value()) {
        ret.push_back(MakeSightEndpoint(*p));
      }
      if (q.has_value()) {
        ret.push_back(MakeSightEndpoint(*q));
      }
    }
  }
  // Remove duplicate endpoints.
  std::sort(ret.begin(), ret.end());
  auto it = std::unique(ret.begin(), ret.end());
  ret.resize(std::distance(ret.begin(), it));
  return ret;
}

std::vector<const Object*> ViewField::NearestK(
    const std::vector<const Object*>& objects, int64_t k) const {
  const int64_t n = objects.size();
  if (n <= k) {
    return objects;
  }
  const Vector2D& o = center();
  std::vector<std::pair<float, const Object*>> dis;
  dis.reserve(n);
  for (const Object* obj : objects) {
    dis.emplace_back(Distance(o, obj->position()), obj);
  }
  std::partial_sort(dis.begin(), dis.begin() + k, dis.end());
  std::vector<const Object*> ret;
  ret.reserve(k);
  for (int64_t i = 0; i < k; ++i) {
    ret.push_back(dis[i].second);
  }
  return ret;
}

}  // namespace nocturne
