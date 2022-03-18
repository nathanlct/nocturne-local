#pragma once

#include <string>

#include "geometry/geometry_utils.h"
#include "geometry/vector_2d.h"
#include "object.h"

namespace nocturne {

class Vehicle : public MovingObject {
 public:
  Vehicle(int64_t id, float length, float width,
          const geometry::Vector2D& position,
          const geometry::Vector2D& destination, float heading, float speed,
          bool can_block_sight, bool can_be_collided, bool check_collision)
      : MovingObject(id, length, width, position, destination, heading, speed,
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
