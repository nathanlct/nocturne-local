#include "traffic_light.h"

#include <SFML/Graphics.hpp>
#include <algorithm>

#include "geometry/vector_2d.h"
#include "utils/sf_utils.h"

namespace nocturne {

geometry::ConvexPolygon TrafficLight::BoundingPolygon() const {
  const geometry::Vector2D p0 =
      position_ + geometry::Vector2D(kTrafficLightRadius, kTrafficLightRadius);
  const geometry::Vector2D p1 =
      position_ + geometry::Vector2D(-kTrafficLightRadius, kTrafficLightRadius);
  const geometry::Vector2D p2 =
      position_ +
      geometry::Vector2D(-kTrafficLightRadius, -kTrafficLightRadius);
  const geometry::Vector2D p3 =
      position_ + geometry::Vector2D(kTrafficLightRadius, -kTrafficLightRadius);
  return geometry::ConvexPolygon({p0, p1, p2, p3});
}

TrafficLightState TrafficLight::LightState() const {
  const auto it =
      std::lower_bound(timestamps_.cbegin(), timestamps_.cend(), current_time_);
  return it == timestamps_.cend()
             ? TrafficLightState::kUnknown
             : light_states_.at(std::distance(timestamps_.cbegin(), it));
}

void TrafficLight::draw(sf::RenderTarget& target,
                        sf::RenderStates states) const {
  const TrafficLightState state = LightState();
  sf::Color color;
  switch (state) {
    case TrafficLightState::kStop: {
      color = sf::Color::Red;
      break;
    }
    case TrafficLightState::kCaution: {
      color = sf::Color::Yellow;
      break;
    }
    case TrafficLightState::kGo: {
      color = sf::Color::Green;
      break;
    }
    case TrafficLightState::kArrowStop: {
      color = sf::Color::Blue;
      break;
    }
    case TrafficLightState::kArrowCaution: {
      color = sf::Color::Magenta;
      break;
    }
    case TrafficLightState::kArrowGo: {
      color = sf::Color::Cyan;
      break;
    }
    case TrafficLightState::kFlashingStop: {
      color = sf::Color{255, 51, 255};
      break;
    }
    case TrafficLightState::kFlashingCaution: {
      color = sf::Color{255, 153, 51};
      break;
    }
    default: {
      // kUnknown
      color = sf::Color{102, 102, 255};
      break;
    }
  }

  constexpr float kRadius = 3.0f;
  sf::CircleShape pentagon(kRadius, 5);
  pentagon.setFillColor(color);
  pentagon.setPosition(utils::ToVector2f(position_));
  target.draw(pentagon, states);
}

TrafficLightState ParseTrafficLightState(const std::string& s) {
  if (s == "stop") {
    return TrafficLightState::kStop;
  } else if (s == "caution") {
    return TrafficLightState::kCaution;
  } else if (s == "go") {
    return TrafficLightState::kGo;
  } else if (s == "arrow_stop") {
    return TrafficLightState::kArrowStop;
  } else if (s == "arrow_caution") {
    return TrafficLightState::kArrowCaution;
  } else if (s == "arrow_go") {
    return TrafficLightState::kArrowGo;
  } else if (s == "flashing_stop") {
    return TrafficLightState::kFlashingStop;
  } else if (s == "flashing_caution") {
    return TrafficLightState::kFlashingCaution;
  } else {
    return TrafficLightState::kUnknown;
  }
}

}  // namespace nocturne
