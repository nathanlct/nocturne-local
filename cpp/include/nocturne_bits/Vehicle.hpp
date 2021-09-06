#pragma once


#include "Object.hpp"
#include <iostream>
#include "Vector2D.hpp"
#include <cmath>


class Vehicle : public Object {
public:
    Vehicle(Vector2D position, float width, float length, float heading);

    void act(float acceleration, float steering);
    virtual void step(float dt);

private:
    void kinematicsUpdate(float dt);
    void dynamicsUpdate(float dt);

    float accelAction;
    float steeringAction;

    float lateralSpeed;
    float yawRate;
};