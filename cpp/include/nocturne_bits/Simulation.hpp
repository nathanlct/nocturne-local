#pragma once

#include "Scenario.hpp"
#include "Object.hpp"
#include "Vector2D.hpp"
#include "utils.hpp"

#include <SFML/Graphics.hpp>

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>

class Simulation {
public:
    Simulation(std::string scenarioPath = "");

    void reset();
    void step(float dt);
    void render();

    void updateView(float padding = 100.0f) const;
    
    void renderCone(Vector2D center, float heading, float viewAngle, const Object* self = nullptr);
    void renderCone(const Object* object, float viewAngle, float headTilt);

    void saveScreenshot();
    Scenario* getScenario() const;

private:
    Scenario* scenario;

    sf::RenderWindow* renderWindow;

    sf::Transform renderTransform;

    sf::Font font;
    sf::Clock clock;
};