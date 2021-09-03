#pragma once

#include <vector>
#include <iostream> // tmp
#include <SFML/Graphics.hpp>

#include "LineType.hpp"
#include "Vector2D.hpp"


class Lane : public sf::Drawable {
public:
    Lane(std::vector<Vector2D> geometry, float width);
    // bool isPointOnLane();

private:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

    float width;
    // float length;

    // LineType leftLineType;
    // LineType rightLineType;

    std::vector<Vector2D> geometry;  // eg 2 points for a straight lane
};

// class StraightLane : public Lane {
//     StraightLane();
// };

// class CircularLane : public Lane {
//     CircularLane();
// };