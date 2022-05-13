#pragma once

#include <SFML/Graphics.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "geometry/vector_2d.h"
#include "object.h"
#include "scenario.h"

namespace nocturne {

class Simulation {
 public:
  Simulation(const std::string& scenario_path = "", int64_t start_time = 0,
             bool use_non_vehicles = true)
      : scenario_path_(scenario_path),
        scenario_(std::make_unique<Scenario>(scenario_path, start_time,
                                             use_non_vehicles)),
        start_time_(start_time),
        use_non_vehicles_(use_non_vehicles) {}

  void Reset() {
    scenario_.reset(
        new Scenario(scenario_path_, start_time_, use_non_vehicles_));
  }

  void Step(float dt) { scenario_->step(dt); }

  void Render();

  Scenario* GetScenario() const { return scenario_.get(); }

  void SaveScreenshot();

 protected:
  void UpdateView(float padding = 100.0f) const;

  const std::string scenario_path_;
  std::unique_ptr<Scenario> scenario_ = nullptr;

  const int64_t start_time_ = 0;
  const bool use_non_vehicles_ = true;

  std::unique_ptr<sf::RenderWindow> render_window_ = nullptr;

  sf::Font font_;
  sf::Clock clock_;
};

}  // namespace nocturne
