#include "Vehicle.hpp"

#include <cmath>

#include "geometry/geometry_utils.h"

namespace nocturne {

void Vehicle::Step(float dt) {
  // if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
  //     accelAction = 100.0f;
  // }
  // else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
  //     if (speed > 0.0f) accelAction = -200.0f;
  //     else accelAction = -100.0f;
  // }
  // else {
  //     if (speed > 0.0f) accelAction = -15.0f;
  //     else if (speed < 0.0f) accelAction = 15.0f;
  //     else accelAction = 0.0f;
  // }

  // if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
  //     steeringAction = -10.0f * pi / 180.0f;
  // }
  // else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
  //     steeringAction = 10.0f * pi / 180.0f;
  // }
  // else {
  //     steeringAction = 0.0f;
  // }

  kinematicsUpdate(dt);
}

void Vehicle::kinematicsUpdate(float dt) {
  float steering = steeringAction;  // todo clip
  float accel = accelAction;        // todo clip

  // kinematic model - table 2.1 from Vehicle Dynamics and Control, R. Rajamani,
  // Chapter 2
  float slipAngle = atan(tan(steering) / 2.0f);
  float dHeading = speed_ * sin(steering) / length_;
  float dX = speed_ * cos(heading_ + slipAngle);
  float dY = speed_ * sin(heading_ + slipAngle);

  heading_ = geometry::utils::AngleAdd(heading_, dHeading * dt);
  position_ += geometry::Vector2D(dX, dY) * dt;
  speed_ += accel * dt;
}

void Vehicle::dynamicsUpdate() {}

}  // namespace nocturne
