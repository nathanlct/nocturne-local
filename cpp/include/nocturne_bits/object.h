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
         const geometry::Vector2D& position, float heading)
      : id_(id),
        length_(length),
        width_(width),
        position_(position),
        heading_(heading) {}

  Object(int64_t id, float length, float width,
         const geometry::Vector2D& position, float heading,
         bool can_block_sight, bool can_be_collided, bool check_collision)
      : id_(id),
        length_(length),
        width_(width),
        position_(position),
        heading_(heading),
        can_block_sight_(can_block_sight),
        can_be_collided_(can_be_collided),
        check_collision_(check_collision) {}

  virtual std::string Type() const { return "Object"; }

  int64_t id() const { return id_; }

  float length() const { return length_; }
  float width() const { return width_; }

  const geometry::Vector2D& position() const { return position_; }
  void set_position(const geometry::Vector2D& position) {
    position_ = position;
  }
  void set_position(float x, float y) { position_ = geometry::Vector2D(x, y); }

  float heading() const { return heading_; }
  void set_heading(float heading) { heading_ = heading; }

  bool can_block_sight() const { return can_block_sight_; }
  bool can_be_collided() const { return can_be_collided_; }
  bool check_collision() const { return check_collision_; }
  bool collided() const { return collided_; }
  void set_collided(bool collided) { collided_ = collided; }

  float Radius() const {
    return std::sqrt(length_ * length_ + width_ * width_) * 0.5f;
  }

  virtual geometry::ConvexPolygon BoundingPolygon() const;

  geometry::AABB GetAABB() const override {
    return BoundingPolygon().GetAABB();
  }

 protected:
  const int64_t id_;
  const float length_ = 0.0f;
  const float width_ = 0.0f;

  geometry::Vector2D position_;
  float heading_ = 0.0f;

  const bool can_block_sight_ = true;
  const bool can_be_collided_ = true;
  const bool check_collision_ = false;
  bool collided_ = false;
};

class MovingObject : public Object {
 public:
  MovingObject() = default;
  MovingObject(int64_t id, float length, float width,
               const geometry::Vector2D& position,
               const geometry::Vector2D& destination, float heading,
               float speed, bool can_block_sight, bool can_be_collided,
               bool check_collision)
      : Object(id, length, width, position, heading, can_block_sight,
               can_be_collided, check_collision),
        destination_(destination),
        speed_(speed),
        random_gen_(std::random_device()()) {
    InitRandomColor();
  }

  std::string Type() const override { return "MovingObject"; }

  float speed() const { return speed_; }
  void set_speed(float speed) { speed_ = speed; }

  const geometry::Vector2D& destination() const { return destination_; }
  void set_destination(const geometry::Vector2D& destination) {
    destination_ = destination;
  }
  void set_destination(float x, float y) {
    destination_ = geometry::Vector2D(x, y);
  }

  const sf::Color& color() const { return color_; }

  sf::RenderTexture* cone_texture() const { return cone_texture_; }
  void set_cone_texture(sf::RenderTexture* cone_texture) {
    cone_texture_ = cone_texture;
  }

  virtual void Step(float dt) = 0;

 protected:
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

  void InitRandomColor();

  geometry::Vector2D destination_;
  float speed_ = 0.0f;

  sf::Color color_;
  sf::RenderTexture* cone_texture_ = nullptr;

  std::mt19937 random_gen_;
};

}  // namespace nocturne
