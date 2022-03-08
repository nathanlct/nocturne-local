#include "Simulation.hpp"

namespace nocturne {

Simulation::Simulation(std::string scenarioFilePath, int startTime,
                       bool useNonVehicles)
    : scenario(new Scenario(scenarioFilePath, startTime, useNonVehicles)),
      scenarioPath(scenarioFilePath),
      renderTransform(),
      renderWindow(nullptr),
      font(),
      clock(),
      startTime(startTime),
      useNonVehicles(useNonVehicles) {
  renderTransform.scale(1, -1);  // horizontal flip
}

void Simulation::step(float dt) { scenario->step(dt); }

void Simulation::waymo_step() { scenario->waymo_step(); }

void Simulation::render() {
  if (renderWindow == nullptr) {
    float winWidth = 1500;
    float winHeight = 800;

    sf::ContextSettings settings;
    settings.antialiasingLevel = 1;
    renderWindow =
        new sf::RenderWindow(sf::VideoMode(winWidth, winHeight), "Nocturne",
                             sf::Style::Default, settings);

    font = nocturne::utils::getFont("Arial.ttf");
  }
  if (renderWindow->isOpen()) {
    sf::Time elapsed = clock.restart();
    float fps = 1.0f / elapsed.asSeconds();

    sf::Event event;
    while (renderWindow->pollEvent(event)) {
      if (event.type == sf::Event::Closed) {
        renderWindow->close();
        // renderWindow = nullptr;
        return;
      } else if (event.type == sf::Event::Resized) {
        updateView();
      }
    }

    renderWindow->clear(sf::Color(50, 50, 50));

    updateView();

    renderWindow->draw(*scenario, renderTransform);

    renderWindow->setView(sf::View(sf::FloatRect(
        0, 0, renderWindow->getSize().x, renderWindow->getSize().y)));

    sf::RectangleShape guiBackgroundTop(
        sf::Vector2f(renderWindow->getSize().x, 35));
    guiBackgroundTop.setPosition(0, 0);
    guiBackgroundTop.setFillColor(sf::Color(0, 0, 0, 100));
    renderWindow->draw(guiBackgroundTop);

    sf::Text text(std::to_string((int)fps) + " fps", font, 20);
    text.setPosition(10, 5);
    text.setFillColor(sf::Color::White);
    renderWindow->draw(text);

    renderWindow->display();
  } else {
    throw std::runtime_error(
        "tried to call the render method but the window is not open.");
  }
}

void Simulation::updateView(float padding) const {
  // TODO(nl) memoize this since its called at every render

  // get rectangle boundaries of the road
  sf::FloatRect scenarioBounds = scenario->getRoadNetworkBoundaries();

  // account for the horizontal flip transform ((x,y) becomes (x,-y))
  scenarioBounds.top = -scenarioBounds.top - scenarioBounds.height;

  // add padding all around
  scenarioBounds.top -= padding;
  scenarioBounds.left -= padding;
  scenarioBounds.width += 2 * padding;
  scenarioBounds.height += 2 * padding;

  // create the view
  sf::Vector2u winSize = renderWindow->getSize();
  sf::Vector2f center =
      sf::Vector2f(scenarioBounds.left + scenarioBounds.width / 2.0f,
                   scenarioBounds.top + scenarioBounds.height / 2.0f);
  sf::Vector2f size = sf::Vector2f(winSize.x, winSize.y) *
                      std::max(scenarioBounds.width / winSize.x,
                               scenarioBounds.height / winSize.y);
  sf::View view(center, size);
  renderWindow->setView(view);
}

void Simulation::reset() {
  scenario = new Scenario(scenarioPath, startTime, useNonVehicles);
}

void Simulation::saveScreenshot() {
  if (renderWindow != nullptr) {
    std::string filename = "./screenshot.png";
    sf::Texture texture;
    texture.create(renderWindow->getSize().x, renderWindow->getSize().y);
    texture.update(*renderWindow);
    texture.copyToImage().saveToFile(filename);
  }
}

Scenario* Simulation::getScenario() const { return scenario; }

}  // namespace nocturne
