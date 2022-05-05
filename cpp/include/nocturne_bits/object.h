#pragma once

#include <SFML/Graphics.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <string>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"
#include "object_base.h"

namespace nocturne {

constexpr float kViewRadius = 120.0f;

enum class ObjectType {
  kUnset = 0,
  kVehicle = 1,
  kPedestrian = 2,
  kCyclist = 3,
  kOther = 4,
};

class Object : public ObjectBase {
 public:
  Object() = default;

  Object(int64_t id, float length, float width,
         const geometry::Vector2D& position,
         const geometry::Vector2D& destination, float heading,
         const geometry::Vector2D& velocity, bool can_block_sight,
         bool can_be_collided, bool check_collision)
      : ObjectBase(position, can_block_sight, can_be_collided, check_collision),
        id_(id),
        length_(length),
        width_(width),
        destination_(destination),
        heading_(heading),
        velocity_(velocity),
        random_gen_(std::random_device()()) {
    InitRandomColor();
  }

  Object(int64_t id, float length, float width,
         const geometry::Vector2D& position,
         const geometry::Vector2D& destination, float heading, float speed,
         bool can_block_sight, bool can_be_collided, bool check_collision)
      : ObjectBase(position, can_block_sight, can_be_collided, check_collision),
        id_(id),
        length_(length),
        width_(width),
        destination_(destination),
        heading_(heading),
        velocity_(geometry::PolarToVector2D(speed, heading)),
        random_gen_(std::random_device()()) {
    InitRandomColor();
  }

  virtual ObjectType Type() const { return ObjectType::kUnset; }

  int64_t id() const { return id_; }

  float length() const { return length_; }
  float width() const { return width_; }

  const geometry::Vector2D& destination() const { return destination_; }
  void set_destination(const geometry::Vector2D& destination) {
    destination_ = destination;
  }
  void set_destination(float x, float y) {
    destination_ = geometry::Vector2D(x, y);
  }

  float heading() const { return heading_; }
  void set_heading(float heading) { heading_ = heading; }

  const geometry::Vector2D& velocity() const { return velocity_; }
  void set_velocity(const geometry::Vector2D& velocity) {
    velocity_ = velocity;
  }
  void set_velocity(float v_x, float v_y) {
    velocity_ = geometry::Vector2D(v_x, v_y);
  }

  float acceleration() const { return acceleration_; }
  void set_acceleration(float acceleration) { acceleration_ = acceleration; }

  float steering() const { return steering_; }
  void set_steering(float steering) { steering_ = steering; }

  bool manual_control() const { return manual_control_; }
  void set_manual_control(bool manual_control) {
    manual_control_ = manual_control;
  }

  bool expert_control() const { return expert_control_; }
  void set_expert_control(bool expert_control) {
    expert_control_ = expert_control;
  }

  const sf::Color& color() const { return color_; }

  sf::RenderTexture* ConeTexture() const { return cone_texture_.get(); }

  bool InitConeTexture(int64_t h, int64_t w,
                       const sf::ContextSettings& settings) {
    if (cone_texture_ == nullptr) {
      cone_texture_ = std::make_unique<sf::RenderTexture>();
      cone_texture_->create(w, h, settings);
      return true;
    }
    return false;
  }

  float Radius() const override {
    return std::sqrt(length_ * length_ + width_ * width_) * 0.5f;
  }

  // Postive for moving forward, negative for moving backward.
  float Speed() const {
    return geometry::DotProduct(velocity_,
                                geometry::PolarToVector2D(1.0f, heading_));
  }

  void SetSpeed(float speed) {
    const float cur_speed = Speed();
    if (geometry::utils::AlmostEquals(cur_speed, 0.0f)) {
      velocity_ = geometry::PolarToVector2D(speed, heading_);
    } else {
      velocity_ *= speed / cur_speed;
    }
  }

  geometry::ConvexPolygon BoundingPolygon() const override;

  void SetActionFromKeyboard();

  virtual void Step(float dt) {
    if (manual_control_) {
      SetActionFromKeyboard();
    }
    KinematicBicycleStep(dt);
  }

 protected:
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

  void InitRandomColor();

  void KinematicBicycleStep(float dt);

  const int64_t id_;

  const float length_ = 0.0f;
  const float width_ = 0.0f;

  geometry::Vector2D destination_;
  float heading_ = 0.0f;
  geometry::Vector2D velocity_;

  float acceleration_ = 0.0f;
  float steering_ = 0.0f;

  // If true the object is controlled by keyboard input.
  bool manual_control_ = false;
  // If true the object is placed along positions in its recorded trajectory.
  bool expert_control_ = false;

  sf::Color color_;
  std::unique_ptr<sf::RenderTexture> cone_texture_ = nullptr;

  std::mt19937 random_gen_;
};

inline ObjectType ParseObjectType(const std::string& type) {
  if (type == "unset") {
    return ObjectType::kUnset;
  } else if (type == "vehicle") {
    return ObjectType::kVehicle;
  } else if (type == "pedestrian") {
    return ObjectType::kPedestrian;
  } else if (type == "cyclist") {
    return ObjectType::kCyclist;
  } else {
    return ObjectType::kOther;
  }
}

}  // namespace nocturne
