#include "geometry/polygon.h"

#include <cmath>
#include <limits>
#include <utility>

#include "geometry/geometry_utils.h"

namespace nocturne {
namespace geometry {

namespace {

bool Separates(const LineSegment& edge, const Polygon& polygon) {
  const Vector2D d = edge.Endpoint1() - edge.Endpoint0();
  for (const Vector2D& p : polygon.vertices()) {
    if (CrossProduct(p - edge.Endpoint0(), d) <= 0.0f) {
      return false;
    }
  }
  return true;
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

// Assume polygon vertices are in counterclockwise order. Check if the other
// polygon lies on the right some one of the edges.
bool ConvexPolygon::Intersects(const ConvexPolygon& polygon) const {
  std::vector<LineSegment> edges = Edges();
  for (const LineSegment& edge : edges) {
    if (Separates(edge, polygon)) {
      return false;
    }
  }
  edges = polygon.Edges();
  for (const LineSegment& edge : edges) {
    if (Separates(edge, *this)) {
      return false;
    }
  }
  return true;
}

}  // namespace geometry
}  // namespace nocturne
