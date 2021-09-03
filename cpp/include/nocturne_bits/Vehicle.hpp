#pragma once


class Object;

class Vehicle : public Object {
public:
    Vehicle();

    // Input is speed or accel, for longitudinal/steering
    // This could be a pedestrian, or an object with a hardcoded movement (move that into Object)
    // Vehicle with bicycle dynamics inherits from this
    void act(float acceleration, float steering);
    void step();

private:
    float width;
    float length;

    float minSpeed;
    float maxSpeed;
};