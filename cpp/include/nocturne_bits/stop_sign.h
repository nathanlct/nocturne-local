#pragma once

#include <SFML/Graphics.hpp>
#include <string>

#include "geometry/aabb.h"
#include "geometry/polygon.h"
#include "object.h"

namespace nocturne {

constexpr float kStopSignRadius = 2.0f;

class StopSign : public Object {
 public:
  StopSign() = default;
  StopSign(int64_t id, const geometry::Vector2D& position)
      : Object(id, /*length=*/kStopSignRadius * 2.0f,
               /*width=*/kStopSignRadius * 2.0f, position,
               /*heading=*/0.0f, /*can_block_sight=*/false,
               /*can_be_collided=*/false, /*check_collision=*/false) {}

  std::string Type() const override { return "StopSign"; }

  geometry::ConvexPolygon BoundingPolygon() const override;

  geometry::AABB GetAABB() const override {
    return geometry::AABB(position_ - kStopSignRadius,
                          position_ + kStopSignRadius);
  }

 protected:
  // TODO: Implement this if needed.
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override {}
};

}  // namespace nocturne
