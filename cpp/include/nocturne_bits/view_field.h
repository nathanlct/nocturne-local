#pragma once

#include <vector>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/circular_sector.h"
#include "geometry/vector_2d.h"
#include "object.h"

namespace nocturne {

class ViewField : public geometry::CircularSector {
 public:
  ViewField() = default;
  ViewField(const geometry::Vector2D& center, float radius, float heading,
            float theta)
      : geometry::CircularSector(center, radius, heading, theta) {}

  std::vector<const Object*> VisibleObjects(
      const std::vector<const Object*>& objects, int64_t limit = -1) const;

  std::vector<const Object*> VisibleNonblockingObjects(
      const std::vector<const Object*>& objects, int64_t limit = -1) const;

  std::vector<const Object*> VisiblePoints(
      const std::vector<const Object*>& objects, int64_t limit = -1) const;

 protected:
  geometry::Vector2D MakeSightEndpoint(const geometry::Vector2D& p) const {
    const geometry::Vector2D& o = center();
    const float r = radius();
    const geometry::Vector2D d = p - o;
    return o + d / d.Norm() * r;
  }

  std::vector<geometry::Vector2D> ComputeSightEndpoints(
      const std::vector<const Object*>& objects) const;

  std::vector<const Object*> NearestK(const std::vector<const Object*>& objects,
                                      int64_t k) const;
};

}  // namespace nocturne
