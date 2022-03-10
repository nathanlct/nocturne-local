#pragma once

#include "Object.hpp"
#include "geometry/geometry_utils.h"
#include "geometry/vector_2d.h"

namespace nocturne {

class Vehicle : public Object {
 public:
  Vehicle(const geometry::Vector2D& position, float width, float length,
          float heading, bool occludes, bool collides, bool checkForCollisions,
          const geometry::Vector2D& goalPosition, int objID, float speed = 0.0);

  void setAccel(float acceleration) { accelAction = acceleration; }

  void setSteeringAngle(float steeringAngle) { steeringAction = steeringAngle; }

  virtual void step(float dt);
  float viewRadius = 120;  // TODO(ev) hardcoding
  std::string type = "Vehicle";

 protected:
  void kinematicsUpdate(float dt);
  void dynamicsUpdate();

  float accelAction;
  float steeringAction;

  float lateralSpeed;
  float yawRate;
};

class Pedestrian : public Vehicle {
 public:
  using Vehicle ::Vehicle;
  std::string type = "Pedestrian";
};

class Cyclist : public Vehicle {
 public:
  using Vehicle ::Vehicle;
  std::string type = "Cyclist";
};

}  // namespace nocturne
