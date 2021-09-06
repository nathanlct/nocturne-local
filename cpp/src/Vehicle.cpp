#include <Vehicle.hpp>


Vehicle::Vehicle(Vector2D position, float width, float length, float heading) :
    Object(position, width, length, heading),
    accelAction(0), steeringAction(0),
    lateralSpeed(0), yawRate(0)
{
    
}

void Vehicle::act(float acceleration, float steering) {
    accelAction = acceleration;
    steeringAction = steering;
}

void Vehicle::step(float dt) {
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
        accelAction = 10.0f;
    }
    else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
        if (speed > 0.0f) accelAction = -30.0f;
        else accelAction = -10.0f;
    }
    else {
        if (speed > 0.0f) accelAction = -5.0f;
        else if (speed < 0.0f) accelAction = 5.0f;
        else accelAction = 0.0f;
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
        steeringAction = -30.0f * pi / 180.0f;
    }
    else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
        steeringAction = 30.0f * pi / 180.0f;
    }
    else {
        steeringAction = 0.0f;
    }
    
    kinematicsUpdate(dt);
}


void Vehicle::kinematicsUpdate(float dt) {
    float steering = steeringAction; // todo clip
    float accel = accelAction; // todo clip

    // kinematic model - table 2.1 from Vehicle Dynamics and Control, R. Rajamani, Chapter 2
    float slipAngle = atan(tan(steering) / 2.0f);
    float dHeading = speed * sin(steering) / length;
    float dX = speed * cos(heading + slipAngle);
    float dY = speed * sin(heading + slipAngle);

    heading += dHeading * dt;
    position += Vector2D(dX, dY) * dt;
    speed += accel * dt;
}

void Vehicle::dynamicsUpdate(float dt) {

}