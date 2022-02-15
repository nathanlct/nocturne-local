#pragma once

#include <SFML/Graphics.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Object.hpp"
#include "Scenario.hpp"
#include "geometry/vector_2d.h"
#include "utils.hpp"

namespace nocturne {

class Simulation {
 public:
  Simulation(std::string scenarioFilePath = "");

  void reset();
  void step(float dt);
  void render();

  void updateView(float padding = 100.0f) const;

  void renderCone(const geometry::Vector2D& center, float heading,
                  float viewAngle, const Object* self = nullptr);
  void renderCone(const Object* object, float viewAngle, float headTilt);

  void saveScreenshot();
  Scenario* getScenario() const;

 private:
  Scenario* scenario;

  sf::RenderWindow* renderWindow;

  sf::Transform renderTransform;

  sf::Font font;
  sf::Clock clock;

  std::string scenarioPath;
};

}  // namespace nocturne
