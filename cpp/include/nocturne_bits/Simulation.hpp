#pragma once

#include "Scenario.hpp"
#include "Object.hpp"
#include "Vector2D.hpp"

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>

class Simulation {
public:
    Simulation(bool render = false, std::string scenarioPath = "");

    void reset();
    void step();

    sf::View getView(sf::Vector2u winSize) const;
    
    void renderCone(Vector2D center, float heading, float viewAngle, const Object* self = nullptr);
    void renderCone(const Object* object, float viewAngle, float headTilt);

private:
    Scenario scenario;
    bool render;

    float circleRadius;
    float renderedCircleRadius;
    sf::RenderTexture circleTexture;

    sf::Transform renderTransform;
};