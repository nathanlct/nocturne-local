#pragma once

#include <vector>
#include <iostream>
#include "Vector2D.hpp"
#include <SFML/Graphics.hpp>

class Point;

class Object : public sf::Drawable {
public:
    Object();

    // bool intersectsWith(Object* other) const; // fast spherical pre-check, then accurate rectangular check
    // std::vector<Point> getCorners() const;

    // void move(); // move according to pos, heading and speed
    virtual void step(float dt);

protected:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

    // bool solid;

    Vector2D position;
    float width;
    float length;
    float heading;

    float speed;
};