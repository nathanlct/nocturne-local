#include "Vehicle.hpp"

#include <cmath>

namespace nocturne {

Vehicle::Vehicle(const geometry::Vector2D& position, float width, float length,
                 float heading, bool occludes, bool collides,
                 bool checkForCollisions,
                 const geometry::Vector2D& goalPosition, int objID, float speed)
    : Object(position, width, length, heading, occludes, collides,
             checkForCollisions, goalPosition, objID, speed),
      accelAction(0),
      steeringAction(0),
      yawRate(0) {}

void Vehicle::step(float dt) {
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
  float dHeading = speed * sin(steering) / length;
  float dX = speed * cos(heading + slipAngle);
  float dY = speed * sin(heading + slipAngle);

  heading += dHeading * dt;
  position += geometry::Vector2D(dX, dY) * dt;
  speed += accel * dt;
}

void Vehicle::dynamicsUpdate() {}

}  // namespace nocturne
