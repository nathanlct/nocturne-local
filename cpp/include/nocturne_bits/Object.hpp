#pragma once

#include <SFML/Graphics.hpp>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"

namespace nocturne {

constexpr float kViewRadius = 120.0f;

class Object : public sf::Drawable, public geometry::AABBInterface {
 public:
  Object() = default;
  Object(int64_t id, float length, float width,
         const geometry::Vector2D& position, float heading, float speed,
         const geometry::Vector2D& goal_position, bool can_block_sight,
         bool can_be_collided, bool check_collision)
      : id_(id),
        length_(length),
        width_(width),
        position_(position),
        heading_(heading),
        speed_(speed),
        goal_position_(goal_position),
        can_block_sight_(can_block_sight),
        can_be_collided_(can_be_collided),
        check_collision_(check_collision),
        random_gen_(std::random_device()()) {
    InitRandomColor();
  }

  virtual std::string Type() const { return "Object"; }
  int64_t id() const { return id_; }

  const geometry::Vector2D& position() const { return position_; }
  void set_position(const geometry::Vector2D& position) {
    position_ = position;
  }

  float length() const { return length_; }
  float width() const { return width_; }

  float heading() const { return heading_; }
  void set_heading(float heading) { heading_ = heading; }

  float speed() const { return speed_; }
  void set_speed(float speed) { speed_ = speed; }

  const geometry::Vector2D& goal_position() const { return goal_position_; }
  void set_goal_position(const geometry::Vector2D& goal_position) {
    goal_position_ = goal_position;
  }

  bool can_block_sight() const { return can_block_sight_; }
  bool can_be_collided() const { return can_be_collided_; }
  bool check_collision() const { return check_collision_; }

  bool collided() const { return collided_; }
  void set_collided(bool collided) { collided_ = collided; }

  const sf::Color& color() const { return color_; }

  sf::RenderTexture* cone_texture() const { return cone_texture_; }
  void set_cone_texture(sf::RenderTexture* cone_texture) {
    cone_texture_ = cone_texture;
  }

  float Radius() const {
    return std::sqrt(length_ * length_ + width_ * width_) * 0.5f;
  }

  geometry::AABB GetAABB() const override {
    return BoundingPolygon().GetAABB();
  }

  geometry::ConvexPolygon BoundingPolygon() const;

  virtual void Step(float /*dt*/) {}

 protected:
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

  void InitRandomColor();

  const int64_t id_;
  const float length_ = 0.0f;
  const float width_ = 0.0f;

  geometry::Vector2D position_;
  float heading_ = 0.0f;
  float speed_ = 0.0f;
  geometry::Vector2D goal_position_;

  const bool can_block_sight_ = false;
  const bool can_be_collided_ = false;
  const bool check_collision_ = false;
  bool collided_ = false;

  sf::Color color_;
  sf::RenderTexture* cone_texture_ = nullptr;

  std::mt19937 random_gen_;
};

}  // namespace nocturne
