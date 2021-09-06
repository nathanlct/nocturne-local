#pragma once

#include "Scenario.hpp"


class Simulation {
public:
    Simulation(bool render);

    void reset();
    void step();

    sf::View getView(sf::Vector2u winSize) const;

    void getCircle() const; // tmp

private:
    Scenario scenario;
    bool render;
};