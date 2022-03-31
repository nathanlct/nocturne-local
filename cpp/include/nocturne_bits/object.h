#pragma once

#include <SFML/Graphics.hpp>
#include <cstdint>
#include <string>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"

namespace nocturne {

constexpr float kViewRadius = 120.0f;

enum class ObjectType {
  kUnset = 0,
  kVehicle = 1,
  kPedestrian = 2,
  kCyclist = 3,
  kRoadPoint = 4,
  kTrafficLight = 5,
  kStopSign = 6,
  kOthers = 7,
};

class Object : public sf::Drawable, public geometry::AABBInterface {
 public:
  Object() = default;

  Object(int64_t id, const geometry::Vector2D& position)
      : id_(id), position_(position) {}

  Object(int64_t id, const geometry::Vector2D& position, bool can_block_sight,
         bool can_be_collided, bool check_collision)
      : id_(id),
        position_(position),
        can_block_sight_(can_block_sight),
        can_be_collided_(can_be_collided),
        check_collision_(check_collision) {}

  virtual ObjectType Type() const { return ObjectType::kUnset; }

  int64_t id() const { return id_; }

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
  const int64_t id_;
  geometry::Vector2D position_;

  const bool can_block_sight_ = false;
  const bool can_be_collided_ = false;
  const bool check_collision_ = false;
  bool collided_ = false;
};

ObjectType ParseObjectType(const std::string& type);

}  // namespace nocturne
