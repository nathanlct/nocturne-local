#pragma once

#include <vector>
#include <SFML/Graphics.hpp>

class Point;

class Object : public sf::Drawable {
public:
    Object();

    bool intersectsWith(Object* other) const; // fast spherical pre-check, then accurate rectangular check
    std::vector<Point> getCorners() const;

    void move(); // move according to pos, heading and speed

protected:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

    bool solid;

    Point pos;  // pos of center (we assume consider rectangular objects for now)
    float width;
    float height;
    float rotation; // or heading

    float velocity;
    // Vector2d speed
    // angular velocity?
};