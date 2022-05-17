#pragma once

#include "geometry/vector_2d.h"
#include "object.h"

namespace nocturne {

class Vehicle : public Object {
 public:
  Vehicle() = default;

  Vehicle(int64_t id, float length, float width,
          const geometry::Vector2D& position,
          const geometry::Vector2D& destination, float heading, float speed,
          bool can_block_sight = true, bool can_be_collided = true,
          bool check_collision = true)
      : Object(id, length, width, position, destination, heading, speed,
               can_block_sight, can_be_collided, check_collision) {}

  Vehicle(int64_t id, float length, float width, float max_speed,
          const geometry::Vector2D& position,
          const geometry::Vector2D& destination, float heading, float speed,
          bool can_block_sight = true, bool can_be_collided = true,
          bool check_collision = true)
      : Object(id, length, width, max_speed, position, destination, heading,
               speed, can_block_sight, can_be_collided, check_collision) {}

  ObjectType Type() const override { return ObjectType::kVehicle; }
};

}  // namespace nocturne
