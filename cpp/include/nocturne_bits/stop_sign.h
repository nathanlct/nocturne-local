#pragma once

#include <SFML/Graphics.hpp>
#include <string>

#include "geometry/aabb.h"
#include "geometry/polygon.h"
#include "object.h"

namespace nocturne {

constexpr float kStopSignRadius = 2.0f;
constexpr int kStopSignNumEdges = 6;

class StopSign : public Object {
 public:
  StopSign() = default;
  StopSign(int64_t id, const geometry::Vector2D& position)
      : Object(id, position,
               /*can_block_sight=*/false,
               /*can_be_collided=*/false, /*check_collision=*/false) {}

  ObjectType Type() const override { return ObjectType::kStopSign; }

  float Radius() const { return kStopSignRadius; }

  geometry::ConvexPolygon BoundingPolygon() const override;

  sf::Color Color() const { return sf::Color::Red; }

 protected:
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
};

}  // namespace nocturne
