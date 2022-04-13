#include "object.h"

namespace nocturne {

ObjectType ParseObjectType(const std::string& type) {
  if (type == "unset") {
    return ObjectType::kUnset;
  } else if (type == "vehicle") {
    return ObjectType::kVehicle;
  } else if (type == "pedestrian") {
    return ObjectType::kPedestrian;
  } else if (type == "cyclist") {
    return ObjectType::kCyclist;
  } else if (type == "road_point") {
    return ObjectType::kRoadPoint;
  } else if (type == "traffic_light") {
    return ObjectType::kTrafficLight;
  } else if (type == "stop_sign") {
    return ObjectType::kStopSign;
  } else {
    return ObjectType::kOthers;
  }
}

}  // namespace nocturne
