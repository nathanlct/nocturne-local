#pragma once

#include "geometry/aabb.h"

namespace nocturne {
namespace geometry {

class AABBInterface {
 public:
  virtual AABB GetAABB() const = 0;
};

}  // namespace geometry
}  // namespace nocturne
