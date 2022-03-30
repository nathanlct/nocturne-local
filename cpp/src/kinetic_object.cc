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

// Kinematic Bicycle Model
// https://thef1clan.com/2020/09/21/vehicle-dynamics-the-kinematic-bicycle-model/
void KineticObject::KinematicBicycleStep(float dt) {
  const float v = Speed();
  const float zeta = steering_;
  const float tan_zeta = std::tan(zeta);
  const float beta = std::atan(tan_zeta * 0.5f);
  const geometry::Vector2D dposition = velocity_.Rotate(beta);
  const float dheading = v * tan_zeta * std::cos(beta) / length_;
  position_ += dposition * dt;
  heading_ = geometry::utils::AngleAdd(heading_, dheading * dt);
  velocity_ = geometry::PolarToVector2D(v + acceleration_, heading_);
}

}  // namespace nocturne
