#pragma once

#include "Road.hpp"
#include "Vector2D.hpp"
#include <vector>
#include <SFML/Graphics.hpp>


class Network : public sf::Drawable {
public:
    Network();

    void addRoad(std::vector<Vector2D> geometry, int lanes, float laneWidth);

    sf::FloatRect getBoundingBox() const;

private:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;
    std::vector<Road> roads;
};