#include "view_field.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#include "geometry/intersection.h"
#include "geometry/line_segment.h"
#include "geometry/polygon.h"
#include "utils/data_utils.h"

namespace nocturne {

namespace {

using geometry::Circle;
using geometry::CircleLike;
using geometry::CircularSector;
using geometry::ConvexPolygon;
using geometry::LineSegment;
using geometry::Vector2D;
using geometry::utils::kTwoPi;

Vector2D MakeSightEndpoint(const CircleLike& vision,
                           const geometry::Vector2D& p) {
  const geometry::Vector2D& o = vision.center();
  const float r = vision.radius();
  const geometry::Vector2D d = p - o;
  return o + d / d.Norm() * r;
}

void VisibleObjectsImpl(const LineSegment& sight,
                        const std::vector<const ObjectBase*>& objects,
                        std::vector<bool>& mask) {
  const int64_t n = objects.size();

  // Compute relative distance from center to each object.
  std::vector<float> dis(n, std::numeric_limits<float>::max());
  float min_dis = std::numeric_limits<float>::max();
  int64_t min_idx = -1;
  for (int64_t i = 0; i < n; ++i) {
    const ObjectBase* obj = objects[i];
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
    const ObjectBase* obj = objects[i];
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

bool IsVisibleNonblockingObject(const CircleLike& vision,
                                const ObjectBase* obj) {
  const auto edges = obj->BoundingPolygon().Edges();
  for (const LineSegment& edge : edges) {
    // Check one endpoint should be enough, the othe one will be checked in
    // the next edge.
    const Vector2D& x = edge.Endpoint0();
    if (vision.Contains(x)) {
      return true;
    }
    const auto [p, q] = vision.Intersection(edge);
    if (p.has_value() || q.has_value()) {
      return true;
    }
  }
  return false;
}

}  // namespace

ViewField::ViewField(const geometry::Vector2D& center, float radius,
                     float heading, float theta)
    : panoramic_view_(theta >= kTwoPi) {
  if (panoramic_view_) {
    vision_ = std::make_unique<Circle>(center, radius);
  } else {
    vision_ = std::make_unique<CircularSector>(center, radius, heading, theta);
  }
}

// O(N^2) algorithm.
// TODO: Implment O(NlogN) algorithm when there are too many objects.
std::vector<const ObjectBase*> ViewField::VisibleObjects(
    const std::vector<const ObjectBase*>& objects) const {
  const int64_t n = objects.size();
  const Vector2D& o = vision_->center();
  const std::vector<Vector2D> sight_endpoints = ComputeSightEndpoints(objects);
  std::vector<bool> mask(n, false);
  for (const Vector2D& p : sight_endpoints) {
    VisibleObjectsImpl(LineSegment(o, p), objects, mask);
  }
  std::vector<const ObjectBase*> ret;
  for (int64_t i = 0; i < n; ++i) {
    if (mask[i]) {
      ret.push_back(objects[i]);
    }
  }
  return ret;
}

void ViewField::FilterVisibleObjects(
    std::vector<const ObjectBase*>& objects) const {
  const int64_t n = objects.size();
  const Vector2D& o = vision_->center();
  const std::vector<Vector2D> sight_endpoints = ComputeSightEndpoints(objects);
  std::vector<bool> mask(n, false);
  for (const Vector2D& p : sight_endpoints) {
    VisibleObjectsImpl(LineSegment(o, p), objects, mask);
  }
  const int64_t pivot = utils::MaskedPartition(mask, objects);
  objects.resize(pivot);
}

std::vector<const ObjectBase*> ViewField::VisibleNonblockingObjects(
    const std::vector<const ObjectBase*>& objects) const {
  std::vector<const ObjectBase*> ret;
  const CircleLike* vptr = vision_.get();
  std::copy_if(objects.cbegin(), objects.cend(), std::back_inserter(ret),
               [vptr](const ObjectBase* o) {
                 return IsVisibleNonblockingObject(*vptr, o);
               });
  return ret;
}

void ViewField::FilterVisibleNonblockingObjects(
    std::vector<const ObjectBase*>& objects) const {
  const CircleLike* vptr = vision_.get();
  auto pivot = std::partition(objects.begin(), objects.end(),
                              [vptr](const ObjectBase* o) {
                                return IsVisibleNonblockingObject(*vptr, o);
                              });
  objects.resize(std::distance(objects.begin(), pivot));
}

std::vector<const geometry::PointLike*> ViewField::VisiblePoints(
    const std::vector<const geometry::PointLike*>& objects) const {
  std::vector<const geometry::PointLike*> ret;
  const std::vector<int32_t> mask = vision_->BatchContains(objects);
  const int64_t n = objects.size();
  for (int64_t i = 0; i < n; ++i) {
    if (mask[i]) {
      ret.push_back(objects[i]);
    }
  }
  return ret;
}

void ViewField::FilterVisiblePoints(
    std::vector<const geometry::PointLike*>& objects) const {
  const std::vector<int32_t> mask = vision_->BatchContains(objects);
  const int64_t pivot = utils::MaskedPartition(mask, objects);
  objects.resize(pivot);
}

std::vector<Vector2D> ViewField::ComputeSightEndpoints(
    const std::vector<const ObjectBase*>& objects) const {
  std::vector<Vector2D> ret;
  const Vector2D& o = vision_->center();
  if (!panoramic_view_) {
    const CircularSector* vptr = dynamic_cast<CircularSector*>(vision_.get());
    ret.push_back(o + vptr->Radius0());
    ret.push_back(o + vptr->Radius1());
  }
  for (const ObjectBase* obj : objects) {
    const auto edges = obj->BoundingPolygon().Edges();
    for (const LineSegment& edge : edges) {
      // Check one endpoint should be enough, the othe one will be checked in
      // the next edge.
      const Vector2D& x = edge.Endpoint0();
      if (vision_->Contains(x)) {
        ret.push_back(MakeSightEndpoint(*vision_, x));
      }
      const auto [p, q] = vision_->Intersection(edge);
      if (p.has_value()) {
        ret.push_back(MakeSightEndpoint(*vision_, *p));
      }
      if (q.has_value()) {
        ret.push_back(MakeSightEndpoint(*vision_, *q));
      }
    }
  }
  // Remove duplicate endpoints.
  std::sort(ret.begin(), ret.end());
  auto it = std::unique(ret.begin(), ret.end());
  ret.resize(std::distance(ret.begin(), it));
  return ret;
}

}  // namespace nocturne
