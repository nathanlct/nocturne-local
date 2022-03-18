#include "stop_sign.h"

#include "geometry/vector_2d.h"

namespace nocturne {

geometry::ConvexPolygon StopSign::BoundingPolygon() const {
  const geometry::Vector2D p0 =
      position_ + geometry::Vector2D(kStopSignRadius, kStopSignRadius);
  const geometry::Vector2D p1 =
      position_ + geometry::Vector2D(-kStopSignRadius, kStopSignRadius);
  const geometry::Vector2D p2 =
      position_ + geometry::Vector2D(-kStopSignRadius, -kStopSignRadius);
  const geometry::Vector2D p3 =
      position_ + geometry::Vector2D(kStopSignRadius, -kStopSignRadius);
  return geometry::ConvexPolygon({p0, p1, p2, p3});
}

}  // namespace nocturne
