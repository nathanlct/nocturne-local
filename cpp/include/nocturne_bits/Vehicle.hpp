#pragma once

#include <string>

#include "Object.hpp"
#include "geometry/geometry_utils.h"
#include "geometry/vector_2d.h"

namespace nocturne {

class Vehicle : public Object {
 public:
  Vehicle(int64_t id, float length, float width,
          const geometry::Vector2D& position, float heading, float speed,
          const geometry::Vector2D& goal_position, bool can_block_sight,
          bool can_be_collided, bool check_collision)
      : Object(id, length, width, position, heading, speed, goal_position,
               can_block_sight, can_be_collided, check_collision) {}

  std::string Type() const override { return "Vehicle"; }

  void setAccel(float acceleration) { accelAction = acceleration; }

  void setSteeringAngle(float steeringAngle) { steeringAction = steeringAngle; }

  void Step(float dt) override;

 protected:
  void kinematicsUpdate(float dt);
  void dynamicsUpdate();

  float accelAction = 0.0f;
  float steeringAction = 0.0f;

  float lateralSpeed;
  float yawRate = 0.0f;
};

// TODO: Make Pedestrian inherited from Object
class Pedestrian : public Vehicle {
 public:
  using Vehicle ::Vehicle;

  std::string Type() const override { return "Pedestrian"; }
};

class Cyclist : public Vehicle {
 public:
  using Vehicle ::Vehicle;

  std::string Type() const override { return "Cyclist"; }
};

}  // namespace nocturne
