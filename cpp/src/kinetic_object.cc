#include "kinetic_object.h"

#include <algorithm>

#include "geometry/geometry_utils.h"
#include "utils/sf_utils.h"

namespace nocturne {

geometry::ConvexPolygon KineticObject::BoundingPolygon() const {
  const geometry::Vector2D p0 =
      geometry::Vector2D(length_ * 0.5f, width_ * 0.5f).Rotate(heading_) +
      position_;
  const geometry::Vector2D p1 =
      geometry::Vector2D(-length_ * 0.5f, width_ * 0.5f).Rotate(heading_) +
      position_;
  const geometry::Vector2D p2 =
      geometry::Vector2D(-length_ * 0.5f, -width_ * 0.5f).Rotate(heading_) +
      position_;
  const geometry::Vector2D p3 =
      geometry::Vector2D(length_ * 0.5f, -width_ * 0.5f).Rotate(heading_) +
      position_;
  return geometry::ConvexPolygon({p0, p1, p2, p3});
}

void KineticObject::draw(sf::RenderTarget& target,
                         sf::RenderStates states) const {
  sf::RectangleShape rect(sf::Vector2f(length_, width_));
  rect.setOrigin(length_ / 2.0f, width_ / 2.0f);
  rect.setPosition(utils::ToVector2f(position_));
  rect.setRotation(geometry::utils::Degrees(heading_));

  sf::Color col;
  if (can_block_sight_ && can_be_collided_) {
    col = color_;
  } else if (can_block_sight_ && !can_be_collided_) {
    col = sf::Color::Blue;
  } else if (!can_block_sight_ && can_be_collided_) {
    col = sf::Color::White;
  } else {
    col = sf::Color::Black;
  }
  rect.setFillColor(col);
  target.draw(rect, states);
}

void KineticObject::InitRandomColor() {
  std::uniform_int_distribution<int32_t> dis(0, 255);
  int32_t r = dis(random_gen_);
  int32_t g = dis(random_gen_);
  int32_t b = dis(random_gen_);
  // Rescale colors to avoid dark objects.
  const int32_t max_rgb = std::max({r, g, b});
  r = r * 255 / max_rgb;
  g = g * 255 / max_rgb;
  b = b * 255 / max_rgb;
  color_ = sf::Color(r, g, b);
}

void KineticObject::SetActionFromKeyboard() {
  // up: accelerate ; down: brake
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
    acceleration_ = 1.5f;
  } else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
    acceleration_ = -3.0f;
  } else {
    // friction
    acceleration_ = -0.5f;
  }

  // right: turn right; left: turn left
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
    steering_ = geometry::utils::Radians(-60.0f);
  } else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
    steering_ = geometry::utils::Radians(60.0f);
  } else {
    steering_ = 0.0f;
  }
}

// Kinematic Bicycle Model
// https://thef1clan.com/2020/09/21/vehicle-dynamics-the-kinematic-bicycle-model/
void KineticObject::KinematicBicycleStep(float dt) {
  const float speed = Speed();
  const float v = speed + acceleration_ * dt * 0.5f;  // Average speed
  const float tan_zeta = std::tan(steering_);
  // Assume center of mass lies at the middle of length, then l / L == 0.5.
  const float beta = std::atan(tan_zeta * 0.5f);
  const geometry::Vector2D d = geometry::PolarToVector2D(v, heading_ + beta);
  const float w = v * tan_zeta * std::cos(beta) / length_;
  position_ += d * dt;
  heading_ = geometry::utils::AngleAdd(heading_, w * dt);
  velocity_ = geometry::PolarToVector2D(speed + acceleration_ * dt, heading_);
}

}  // namespace nocturne
