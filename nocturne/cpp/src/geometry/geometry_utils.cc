#include "geometry/geometry_utils.h"

#include "geometry/point_like.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {
namespace utils {

std::pair<std::vector<float>, std::vector<float>> PackCoordinates(
    const std::vector<Vector2D>& points) {
  const int64_t n = points.size();
  std::vector<float> x(n);
  std::vector<float> y(n);
  for (int64_t i = 0; i < n; ++i) {
    x[i] = points[i].x();
    y[i] = points[i].y();
  }
  return std::make_pair(x, y);
}

std::pair<std::vector<float>, std::vector<float>> PackCoordinates(
    const std::vector<const PointLike*>& points) {
  const int64_t n = points.size();
  std::vector<float> x(n);
  std::vector<float> y(n);
  for (int64_t i = 0; i < n; ++i) {
    const Vector2D p = points[i]->Coordinate();
    x[i] = p.x();
    y[i] = p.y();
  }
  return std::make_pair(x, y);
}

}  // namespace utils
}  // namespace geometry
}  // namespace nocturne
