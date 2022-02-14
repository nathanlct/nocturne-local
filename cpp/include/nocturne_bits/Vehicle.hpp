#pragma once

#include "Object.hpp"
#include "geometry/vector_2d.h"

namespace nocturne {

class Vehicle : public Object {
 public:
  Vehicle(const geometry::Vector2D& position, float width, float length,
          float heading, bool occludes, bool collides, bool checkForCollisions,
          const geometry::Vector2D& goalPosition);

  void setAccel(float acceleration) { accelAction = acceleration; }

  void setSteeringAngle(float steeringAngle) { steeringAction = steeringAngle; }

  virtual void step(float dt);

 private:
  void kinematicsUpdate(float dt);
  void dynamicsUpdate(float dt);

  float accelAction;
  float steeringAction;

  float lateralSpeed;
  float yawRate;
};

}  // namespace nocturne
