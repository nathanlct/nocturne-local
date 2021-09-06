#pragma once

#include <vector>
#include <iostream>
#include <cmath>

#include <SFML/Graphics.hpp>
#include "Vector2D.hpp"
#include "LineType.hpp"


class Road : public sf::Drawable {
public:
    Road(std::vector<Vector2D> geometry, int lanes, float laneWidth);


private:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

    void buildLanes();
    void buildRoadGraphics();

    std::vector<Vector2D> geometry;
    float laneWidth;
    int nLanes;

    float initialAngleDelta;
    float finalAngleDelta;
    std::vector<float> angles;
    std::vector<float> anglesDelta;

    std::vector<std::vector<Vector2D>> lanesGeometry;

    std::vector<sf::ConvexShape> laneQuads;
    std::vector<sf::Vertex> roadLines;
};