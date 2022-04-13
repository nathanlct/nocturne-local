#pragma once

#include "geometry/vector_2d.h"
#include "kinetic_object.h"

namespace nocturne {

class Pedestrian : public KineticObject {
 public:
  Pedestrian() = default;

  Pedestrian(int64_t id, float length, float width,
             const geometry::Vector2D& position,
             const geometry::Vector2D& destination, float heading,
             const geometry::Vector2D& velocity, bool can_block_sight,
             bool can_be_collided, bool check_collision)
      : KineticObject(id, length, width, position, destination, heading,
                      velocity, can_block_sight, can_be_collided,
                      check_collision) {}

  Pedestrian(int64_t id, float length, float width,
             const geometry::Vector2D& position,
             const geometry::Vector2D& destination, float heading, float speed,
             bool can_block_sight, bool can_be_collided, bool check_collision)
      : KineticObject(id, length, width, position, destination, heading, speed,
                      can_block_sight, can_be_collided, check_collision) {}

  ObjectType Type() const override { return ObjectType::kPedestrian; }
};

}  // namespace nocturne
