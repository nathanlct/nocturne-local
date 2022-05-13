#include "simulation.h"

#include "utils/sf_utils.h"

namespace nocturne {

void Simulation::Render() {
  if (render_window_ == nullptr) {
    constexpr int64_t kWinWidth = 1500;
    constexpr int64_t kWinHeight = 800;

    sf::ContextSettings settings;
    settings.antialiasingLevel = 1;
    render_window_ = std::make_unique<sf::RenderWindow>(
        sf::VideoMode(kWinWidth, kWinHeight), "Nocturne", sf::Style::Default,
        settings);
    font_ = utils::LoadFont("Arial.ttf");
  }
  if (render_window_->isOpen()) {
    sf::Time elapsed = clock_.restart();
    float fps = 1.0f / elapsed.asSeconds();

    sf::Event event;
    while (render_window_->pollEvent(event)) {
      if (event.type == sf::Event::Closed) {
        render_window_->close();
        return;
      } else if (event.type == sf::Event::Resized) {
        UpdateView();
      }
    }

    render_window_->clear(sf::Color(50, 50, 50));
    UpdateView();
    render_window_->draw(*scenario_, render_transform_);
    render_window_->setView(sf::View(sf::FloatRect(
        0, 0, render_window_->getSize().x, render_window_->getSize().y)));

    sf::RectangleShape guiBackgroundTop(
        sf::Vector2f(render_window_->getSize().x, 35));
    guiBackgroundTop.setPosition(0, 0);
    guiBackgroundTop.setFillColor(sf::Color(0, 0, 0, 100));
    render_window_->draw(guiBackgroundTop);

    sf::Text text(std::to_string((int)fps) + " fps", font_, 20);
    text.setPosition(10, 5);
    text.setFillColor(sf::Color::White);
    render_window_->draw(text);

    render_window_->display();
  } else {
    throw std::runtime_error(
        "tried to call the render method but the window is not open.");
  }
}

void Simulation::SaveScreenshot() {
  if (render_window_ != nullptr) {
    const std::string filename = "./screenshot.png";
    sf::Texture texture;
    texture.create(render_window_->getSize().x, render_window_->getSize().y);
    texture.update(*render_window_);
    texture.copyToImage().saveToFile(filename);
  }
}

void Simulation::UpdateView(float padding) const {
  const sf::Vector2u win_size = render_window_->getSize();
  const sf::View scenario_view = scenario_->View(win_size.x, win_size.y, padding);
  render_window_->setView(scenario_view);
}

}  // namespace nocturne
