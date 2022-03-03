#include "geometry/polygon.h"

#include <cmath>
#include <limits>
#include <utility>

#include "geometry/geometry_utils.h"

namespace nocturne {
namespace geometry {

namespace {

std::pair<float, float> MinMaxProjection(const Polygon& polygon,
                                         const Vector2D& normal_vec) {
  float min_proj = std::numeric_limits<float>::max();
  float max_proj = std::numeric_limits<float>::lowest();
  for (const Vector2D& p : polygon.vertices()) {
    const float proj = DotProduct(normal_vec, p);
    min_proj = std::min(min_proj, proj);
    max_proj = std::max(max_proj, proj);
  }
  return std::make_pair(min_proj, max_proj);
}

bool Separates(const Polygon& a, const Polygon& b, const Vector2D& normal_vec) {
  const auto [min1, max1] = MinMaxProjection(a, normal_vec);
  const auto [min2, max2] = MinMaxProjection(b, normal_vec);
  return max1 < min2 || max2 < min1;
}

}  // namespace

AABB Polygon::GetAABB() const {
  float min_x = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float min_y = std::numeric_limits<float>::max();
  float max_y = std::numeric_limits<float>::lowest();
  for (const Vector2D& v : vertices_) {
    min_x = std::min(min_x, v.x());
    max_x = std::max(max_x, v.x());
    min_y = std::min(min_y, v.y());
    max_y = std::max(max_y, v.y());
  }
  return AABB(min_x, min_y, max_x, max_y);
}

std::vector<LineSegment> Polygon::Edges() const {
  std::vector<LineSegment> edges;
  const int64_t n = vertices_.size();
  edges.reserve(n);
  for (int64_t i = 1; i < n; ++i) {
    edges.emplace_back(vertices_[i - 1], vertices_[i]);
  }
  edges.emplace_back(vertices_.back(), vertices_.front());
  return edges;
}

float Polygon::Area() const {
  const int64_t n = vertices_.size();
  float s = CrossProduct(vertices_.back(), vertices_.front());
  for (int64_t i = 1; i < n; ++i) {
    s += CrossProduct(vertices_[i - 1], vertices_[i]);
  }
  return std::fabs(s) * 0.5f;
}

// Time Complexy: O(N)
// TODO: Add O(logN) algorithm if some polygon contains many vertices.
bool ConvexPolygon::Contains(const Vector2D& p) const {
  const int64_t n = vertices_.size();
  float s =
      std::fabs(CrossProduct(vertices_.back() - p, vertices_.front() - p));
  for (int64_t i = 1; i < n; ++i) {
    s += std::fabs(CrossProduct(vertices_[i - 1] - p, vertices_[i] - p));
  }
  // return s * 0.5f == Area();
  return utils::AlmostEquals(s * 0.5f, Area());
}

bool ConvexPolygon::Intersects(const ConvexPolygon& polygon) const {
  std::vector<LineSegment> edges = Edges();
  for (const LineSegment& edge : edges) {
    if (Separates(*this, polygon, edge.NormalVector())) {
      return false;
    }
  }
  edges = polygon.Edges();
  for (const LineSegment& edge : edges) {
    if (Separates(*this, polygon, edge.NormalVector())) {
      return false;
    }
  }
  return true;
}

}  // namespace geometry
}  // namespace nocturne
