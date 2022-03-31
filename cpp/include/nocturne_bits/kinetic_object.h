#pragma once

#include <SFML/Graphics.hpp>
#include <cmath>
#include <cstdint>
#include <random>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"
#include "object.h"

namespace nocturne {

class KineticObject : public Object {
 public:
  KineticObject() = default;

  KineticObject(int64_t id, float length, float width,
                const geometry::Vector2D& position,
                const geometry::Vector2D& destination, float heading,
                const geometry::Vector2D& velocity, bool can_block_sight,
                bool can_be_collided, bool check_collision)
      : Object(id, position, can_block_sight, can_be_collided, check_collision),
        length_(length),
        width_(width),
        destination_(destination),
        heading_(heading),
        velocity_(velocity),
        random_gen_(std::random_device()()) {
    InitRandomColor();
  }

  KineticObject(int64_t id, float length, float width,
                const geometry::Vector2D& position,
                const geometry::Vector2D& destination, float heading,
                float speed, bool can_block_sight, bool can_be_collided,
                bool check_collision)
      : Object(id, position, can_block_sight, can_be_collided, check_collision),
        length_(length),
        width_(width),
        destination_(destination),
        heading_(heading),
        velocity_(geometry::PolarToVector2D(speed, heading)),
        random_gen_(std::random_device()()) {
    InitRandomColor();
  }

  float Radius() const override {
    return std::sqrt(length_ * length_ + width_ * width_) * 0.5f;
  }

  geometry::ConvexPolygon BoundingPolygon() const override;

  float length() const { return length_; }
  float width() const { return width_; }

  float heading() const { return heading_; }
  void set_heading(float heading) { heading_ = heading; }

  const geometry::Vector2D& velocity() const { return velocity_; }
  void set_velocity(const geometry::Vector2D& velocity) {
    velocity_ = velocity;
  }
  void set_velocity(float v_x, float v_y) {
    velocity_ = geometry::Vector2D(v_x, v_y);
  }

  float Speed() const { return velocity_.Norm(); }
  void SetSpeed(float speed) {
    const float cur_speed = Speed();
    if (geometry::utils::AlmostEquals(cur_speed, 0.0f)) {
      velocity_ = geometry::PolarToVector2D(speed, heading_);
    } else {
      velocity_ *= speed / cur_speed;
    }
  }

  const geometry::Vector2D& destination() const { return destination_; }
  void set_destination(const geometry::Vector2D& destination) {
    destination_ = destination;
  }
  void set_destination(float x, float y) {
    destination_ = geometry::Vector2D(x, y);
  }

  float keyboard_controllable() const { return keyboard_controllable_; }
  void set_keyboard_controllable(bool keyboard_controllable) {
    keyboard_controllable_ = keyboard_controllable;
  }

  float acceleration() const { return acceleration_; }
  void set_acceleration(float acceleration) { acceleration_ = acceleration; }

  float steering() const { return steering_; }
  void set_steering(float steering) { steering_ = steering; }

  const sf::Color& color() const { return color_; }

  sf::RenderTexture* cone_texture() const { return cone_texture_; }
  void set_cone_texture(sf::RenderTexture* cone_texture) {
    cone_texture_ = cone_texture;
  }

  void SetActionFromKeyboard();

  virtual void Step(float dt) {
    if (keyboard_controllable_) {
      SetActionFromKeyboard();
    }
    KinematicBicycleStep(dt);
  }

 protected:
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

  void InitRandomColor();

  void KinematicBicycleStep(float dt);

  const float length_ = 0.0f;
  const float width_ = 0.0f;

  geometry::Vector2D destination_;
  float heading_ = 0.0f;
  geometry::Vector2D velocity_;

  float acceleration_ = 0.0f;
  float steering_ = 0.0f;

  bool keyboard_controllable_ = false;

  sf::Color color_;
  sf::RenderTexture* cone_texture_ = nullptr;

  std::mt19937 random_gen_;
};

}  // namespace nocturne
