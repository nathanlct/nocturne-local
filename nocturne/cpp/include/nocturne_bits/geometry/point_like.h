#pragma once

#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

class PointLike {
 public:
  virtual Vector2D Coordinate() const = 0;
};

}  // namespace geometry
}  // namespace nocturne
