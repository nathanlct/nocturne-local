#pragma once

#include <SFML/Graphics.hpp>
#include <cstdint>
#include <string>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"

namespace nocturne {

class ObjectBase : public sf::Drawable, public geometry::AABBInterface {
 public:
  ObjectBase() = default;

  explicit ObjectBase(const geometry::Vector2D& position)
      : position_(position) {}

  ObjectBase(const geometry::Vector2D& position, bool can_block_sight,
             bool can_be_collided, bool check_collision)
      : position_(position),
        can_block_sight_(can_block_sight),
        can_be_collided_(can_be_collided),
        check_collision_(check_collision) {}

  const geometry::Vector2D& position() const { return position_; }
  void set_position(const geometry::Vector2D& position) {
    position_ = position;
  }
  void set_position(float x, float y) { position_ = geometry::Vector2D(x, y); }

  bool can_block_sight() const { return can_block_sight_; }
  bool can_be_collided() const { return can_be_collided_; }
  bool check_collision() const { return check_collision_; }
  bool collided() const { return collided_; }
  void set_collided(bool collided) { collided_ = collided; }

  virtual float Radius() const = 0;

  virtual geometry::ConvexPolygon BoundingPolygon() const = 0;

  geometry::AABB GetAABB() const override {
    return BoundingPolygon().GetAABB();
  }

 protected:
  geometry::Vector2D position_;

  const bool can_block_sight_ = false;
  const bool can_be_collided_ = false;
  const bool check_collision_ = false;
  bool collided_ = false;
};

}  // namespace nocturne
