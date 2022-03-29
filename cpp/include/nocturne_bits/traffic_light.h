#pragma once

#include <SFML/Graphics.hpp>
#include <cassert>
#include <string>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/polygon.h"
#include "object.h"

namespace nocturne {

constexpr float kTrafficLightRadius = 2.0f;
constexpr int kTrafficLightNumEdges = 5;

enum class TrafficLightState {
  kUnknown = 0,
  kStop = 1,
  kCaution = 2,
  kGo = 3,
  kArrowStop = 4,
  kArrowCaution = 5,
  kArrowGo = 6,
  kFlashingStop = 7,
  kFlashingCaution = 8,
};

class TrafficLight : public Object {
 public:
  TrafficLight() = default;
  TrafficLight(int64_t id, const geometry::Vector2D& position,
               const std::vector<int64_t> timestamps,
               const std::vector<TrafficLightState>& light_states,
               int64_t current_time)
      : Object(id, position,
               /*can_block_sight=*/false,
               /*can_be_collided=*/false, /*check_collision=*/false),
        timestamps_(timestamps),
        light_states_(light_states),
        current_time_(current_time) {
    assert(timestamps_.size() == light_states_.size());
  }

  ObjectType Type() const override { return ObjectType::kTrafficLight; }

  float Radius() const { return kTrafficLightRadius; }

  geometry::ConvexPolygon BoundingPolygon() const override;

  const std::vector<int64_t>& timestamps() const { return timestamps_; }
  const std::vector<TrafficLightState>& light_states() const {
    return light_states_;
  }

  int64_t current_time() const { return current_time_; }
  void set_current_time(int64_t current_time) { current_time_ = current_time; }

  TrafficLightState LightState() const;

 protected:
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

  const std::vector<int64_t> timestamps_;
  const std::vector<TrafficLightState> light_states_;
  int64_t current_time_;
};

TrafficLightState ParseTrafficLightState(const std::string& s);

}  // namespace nocturne
