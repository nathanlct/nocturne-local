#pragma once

#include "geometry/vector_2d.h"
#include "object_base.h"

namespace nocturne {

enum class StaticObjectType {
  kUnset = 0,
  kRoadPoint = 1,
  kTrafficLight = 2,
  kStopSign = 3,
  kOther = 4,
};

class StaticObject : public ObjectBase {
 public:
  StaticObject() = default;
  explicit StaticObject(const geometry::Vector2D& position)
      : ObjectBase(position) {}
  StaticObject(const geometry::Vector2D& position, bool can_block_sight,
               bool can_be_collided, bool check_collision)
      : ObjectBase(position, can_block_sight, can_be_collided,
                   check_collision) {}

  virtual StaticObjectType Type() const { return StaticObjectType::kUnset; }
};

inline StaticObjectType ParseStaticObjectType(const std::string& type) {
  if (type == "unset") {
    return StaticObjectType::kUnset;
  } else if (type == "road_point") {
    return StaticObjectType::kRoadPoint;
  } else if (type == "traffic_light") {
    return StaticObjectType::kTrafficLight;
  } else if (type == "stop_sign") {
    return StaticObjectType::kStopSign;
  } else {
    return StaticObjectType::kOther;
  }
}

}  // namespace nocturne
