#pragma once

#include <initializer_list>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/line_segment.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

class Polygon : public AABBInterface {
 public:
  Polygon() = default;
  explicit Polygon(const std::initializer_list<Vector2D>& vertices)
      : vertices_(vertices) {}
  explicit Polygon(const std::vector<Vector2D>& vertices)
      : vertices_(vertices) {}
  explicit Polygon(std::vector<Vector2D>&& vertices) : vertices_(vertices) {}

  AABB GetAABB() const override;

  int64_t NumEdges() const { return vertices_.size(); }

  const std::vector<Vector2D>& vertices() const { return vertices_; }
  const Vector2D& Vertex(int64_t index) const { return vertices_.at(index); }

  std::vector<LineSegment> Edges() const;

  float Area() const;

 protected:
  std::vector<Vector2D> vertices_;
};

class ConvexPolygon : public Polygon {
 public:
  ConvexPolygon() = default;
  explicit ConvexPolygon(const std::initializer_list<Vector2D>& vertices)
      : Polygon(vertices) {}
  explicit ConvexPolygon(const std::vector<Vector2D>& vertices)
      : Polygon(vertices) {}
  explicit ConvexPolygon(std::vector<Vector2D>&& vertices)
      : Polygon(vertices) {}

  bool Contains(const Vector2D& p) const;

  bool Intersects(const ConvexPolygon& polygon) const;
};

}  // namespace geometry
}  // namespace nocturne
