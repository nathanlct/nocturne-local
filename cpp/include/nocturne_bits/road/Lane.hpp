#pragma once

#include <vector>
#include <SFML/Graphics.hpp>

class Point;
class LineType;


class Lane : public sf::Drawable {
public:
    Lane();
    bool isPointOnLane();

private:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

    float width;
    float length;

    LineType leftLineType;
    LineType rightLineType;

    std::vector<Point> geometry;  // eg 2 points for a straight lane
};

class StraightLane : public Lane {
    Lane();
};

class CircularLane : public Lane {
    Lane();
};