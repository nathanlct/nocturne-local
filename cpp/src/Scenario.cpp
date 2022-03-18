#include "Scenario.hpp"

#include <algorithm>
#include <limits>
#include <memory>

#include "geometry/aabb_interface.h"
#include "geometry/geometry_utils.h"
#include "geometry/intersection.h"
#include "geometry/line_segment.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"
#include "utils/sf_utils.h"

namespace nocturne {

using geometry::utils::kPi;
Scenario::Scenario(std::string path, int startTime, bool useNonVehicles)
    : currTime(startTime),
      useNonVehicles(useNonVehicles),
      lineSegments(),
      roadLines(),
      vehicles(),
      pedestrians(),
      cyclists(),
      roadObjects(),
      stopSigns(),
      trafficLights(),
      imageTexture(nullptr),
      expertTrajectories(),
      expertSpeeds(),
      expertHeadings(),
      lengths(),
      expertValid() {
  if (path.size() > 0) {
    loadScenario(path);
  } else {
    throw std::invalid_argument("No scenario file inputted.");
    // TODO(nl) right now an empty scenario crashes, expectedly
    std::cout << "No scenario path inputted. Defaulting to an empty scenario."
              << std::endl;
  }
}

void Scenario::loadScenario(std::string path) {
  std::ifstream data(path);
  if (!data.is_open()) {
    throw std::invalid_argument("Scenario file couldn't be opened: " + path);
  }
  json j;
  data >> j;

  name = j["name"];

  for (const auto& obj : j["objects"]) {
    std::string type = obj["type"];
    // TODO(ev) currTime should be passed in rather than defined here
    geometry::Vector2D pos(obj["position"]["x"][currTime],
                           obj["position"]["y"][currTime]);
    float width = obj["width"];
    float length = obj["length"];
    float heading = geometry::utils::NormalizeAngle(
        geometry::utils::Radians(static_cast<float>(obj["heading"][currTime])));

    // TODO(ev) this should be set elsewhere
    bool occludes = true;
    bool collides = true;
    bool checkForCollisions = true;

    geometry::Vector2D goalPos;
    if (obj.contains("goalPosition")) {
      goalPos = geometry::Vector2D(obj["goalPosition"]["x"],
                                   obj["goalPosition"]["y"]);
    }
    std::vector<geometry::Vector2D> localExpertTrajectory;
    std::vector<geometry::Vector2D> localExpertSpeeds;
    std::vector<bool> localValid;
    std::vector<float> localHeadingVec;
    bool didObjectMove = false;
    for (unsigned int i = 0; i < obj["position"]["x"].size(); i++) {
      geometry::Vector2D currPos(obj["position"]["x"][i],
                                 obj["position"]["y"][i]);
      geometry::Vector2D currVel(obj["velocity"]["x"][i],
                                 obj["velocity"]["y"][i]);
      localExpertTrajectory.push_back(currPos);
      localExpertSpeeds.push_back(currVel);
      if (currVel.Norm() > 0) {
        didObjectMove = true;
      }
      localValid.push_back(bool(obj["valid"][i]));
      // waymo data is in degrees!
      float expertHeading = geometry::utils::NormalizeAngle(
          geometry::utils::Radians(float(obj["heading"][i])));
      localHeadingVec.push_back(expertHeading);
    }
    // TODO(ev) make it a flag whether all vehicles are added or just the
    // vehicles that are valid

    // we only want to store and load vehicles that are valid at this
    // initialization time
    if (bool(obj["valid"][currTime])) {
      expertTrajectories.push_back(localExpertTrajectory);
      expertSpeeds.push_back(localExpertSpeeds);
      expertHeadings.push_back(localHeadingVec);
      lengths.push_back(length);
      expertValid.push_back(localValid);
      if (type == "vehicle") {
        std::shared_ptr<Vehicle> vehicle = std::make_shared<Vehicle>(
            IDCounter, length, width, pos, goalPos, heading,
            localExpertSpeeds[currTime].Norm(), occludes, collides,
            checkForCollisions);
        vehicles.push_back(vehicle);
        roadObjects.push_back(vehicle);
        if (didObjectMove) {
          objectsThatMoved.push_back(vehicle);
        }
      } else if (type == "pedestrian" && useNonVehicles) {
        std::shared_ptr<Pedestrian> pedestrian = std::make_shared<Pedestrian>(
            IDCounter, length, width, pos, goalPos, heading,
            localExpertSpeeds[currTime].Norm(), occludes, collides,
            checkForCollisions);
        pedestrians.push_back(pedestrian);
        roadObjects.push_back(pedestrian);
        if (didObjectMove) {
          objectsThatMoved.push_back(pedestrian);
        }
      } else if (type == "cyclist" && useNonVehicles) {
        std::shared_ptr<Cyclist> cyclist = std::make_shared<Cyclist>(
            IDCounter, length, width, pos, goalPos, heading,
            localExpertSpeeds[currTime].Norm(), occludes, collides,
            checkForCollisions);
        cyclists.push_back(cyclist);
        roadObjects.push_back(cyclist);
        if (didObjectMove) {
          objectsThatMoved.push_back(cyclist);
        }
      }
      // No point in printing this if we are not using non-vehicles
      // TODO(ev) we should include the UNKNOWN type objects
      else if (useNonVehicles) {
        std::cerr << "Unknown object type: " << type << std::endl;
      }
      IDCounter++;
    }
  }

  float minX, minY, maxX, maxY;
  bool first = true;

  // TODO(ev) refactor into a roadline object
  for (const auto& road : j["roads"]) {
    std::vector<geometry::Vector2D> laneGeometry;
    std::string type = road["type"];
    const RoadType road_type = ParseRoadType(type);
    const bool checkForCollisions = (road_type == RoadType::kRoadEdge);
    // // TODO(ev) this is not good
    // if (type == "lane") {
    //   road_type = RoadType::lane;
    // } else if (type == "road_line") {
    //   road_type = RoadType::road_line;
    // } else if (type == "road_edge") {
    //   road_type = RoadType::road_edge;
    //   checkForCollisions = true;
    // } else if (type == "stop_sign") {
    //   road_type = RoadType::stop_sign;
    // } else if (type == "crosswalk") {
    //   road_type = RoadType::crosswalk;
    // } else if (type == "speed_bump") {
    //   road_type = RoadType::speed_bump;
    // } else {
    //   road_type = RoadType::none;
    // }
    // we have to handle stop signs differently from other lane types
    if (road_type == RoadType::kStopSign) {
      const geometry::Vector2D position(road["geometry"][0]["x"],
                                        road["geometry"][0]["y"]);
      std::shared_ptr<StopSign> stop_sign =
          std::make_shared<StopSign>(stopSigns.size(), position);
      stopSigns.emplace_back(stop_sign);
    } else {
      // Iterate over every line segment
      for (size_t i = 0; i < road["geometry"].size() - 1; i++) {
        laneGeometry.emplace_back(road["geometry"][i]["x"],
                                  road["geometry"][i]["y"]);
        geometry::Vector2D startPoint(road["geometry"][i]["x"],
                                      road["geometry"][i]["y"]);
        geometry::Vector2D endPoint(road["geometry"][i + 1]["x"],
                                    road["geometry"][i + 1]["y"]);
        // track the minimum boundaries
        if (first) {
          minX = maxX = startPoint.x();
          minY = maxY = startPoint.y();
          first = false;
        } else {
          minX = std::min(minX, std::min(startPoint.x(), endPoint.x()));
          maxX = std::max(maxX, std::max(startPoint.x(), endPoint.x()));
          minY = std::min(minY, std::min(startPoint.y(), endPoint.y()));
          maxY = std::max(maxY, std::max(startPoint.y(), endPoint.y()));
        }
        // We only want to store the line segments for collision checking if
        // collisions are possible
        if (checkForCollisions == true) {
          std::shared_ptr<geometry::LineSegment> lineSegment =
              std::make_shared<geometry::LineSegment>(startPoint, endPoint);
          lineSegments.push_back(lineSegment);
        }
      }
    }
    // Now construct the entire roadline object which is what will be used for
    // drawing
    int road_size = road["geometry"].size();
    laneGeometry.emplace_back(road["geometry"][road_size - 1]["x"],
                              road["geometry"][road_size - 1]["y"]);
    std::shared_ptr<RoadLine> roadLine =
        std::make_shared<RoadLine>(road_type, std::move(laneGeometry),
                                   /*num_road_points=*/8, checkForCollisions);
    roadLines.push_back(roadLine);
  }
  roadNetworkBounds = sf::FloatRect(minX, minY, maxX - minX, maxY - minY);

  // Now create the BVH for the line segments
  // Since the line segments never move we only need to define this once
  const int64_t n = lineSegments.size();
  std::vector<const geometry::AABBInterface*> objects;
  objects.reserve(n);
  for (const auto& obj : lineSegments) {
    objects.push_back(dynamic_cast<const geometry::AABBInterface*>(obj.get()));
  }
  line_segment_bvh_.InitHierarchy(objects);

  // Now handle the traffic light states
  for (const auto& tl : j["tl_states"]) {
    maxEnvTime = 90;
    // Lane positions don't move so we can just use the first
    // element
    float x_pos = float(tl["x"][0]);
    float y_pos = float(tl["y"][0]);
    std::vector<int64_t> validTimes;
    std::vector<TrafficLightState> lightStates;

    for (size_t i = 0; i < tl["state"].size(); i++) {
      // TODO(ev) do this more compactly
      const TrafficLightState light_state =
          ParseTrafficLightState(tl["state"][i]);
      lightStates.push_back(light_state);
      validTimes.push_back(int(tl["time_index"][i]));
    }
    std::shared_ptr<TrafficLight> traffic_light =
        std::make_shared<TrafficLight>(trafficLights.size(),
                                       geometry::Vector2D(x_pos, y_pos),
                                       validTimes, lightStates, currTime);
    trafficLights.push_back(traffic_light);
  }

  // initialize the road objects bvh
  const int64_t nRoadObjects = roadObjects.size();
  if (nRoadObjects > 0) {
    std::vector<const geometry::AABBInterface*> storeObjects;
    storeObjects.reserve(nRoadObjects);
    for (const auto& obj : roadObjects) {
      storeObjects.push_back(
          dynamic_cast<const geometry::AABBInterface*>(obj.get()));
    }
    vehicle_bvh_.InitHierarchy(storeObjects);
  }

  // Now create the BVH for the inividual road points that will be passed to our
  // agent as state Since the line segments never move we only need to define
  // this once
  const int64_t nLines = roadLines.size();
  const int64_t nPointsPerLine = roadLines[0]->num_road_points();
  if ((nLines * nPointsPerLine) > 0) {
    std::vector<const geometry::AABBInterface*> roadPointObjects;
    roadPointObjects.reserve(nLines * nPointsPerLine);
    for (const auto& roadLine : roadLines) {
      for (const auto& roadPoint : roadLine->road_points()) {
        roadPointObjects.push_back(
            dynamic_cast<const geometry::AABBInterface*>(&roadPoint));
      }
    }
    road_point_bvh.InitHierarchy(roadPointObjects);
  }

  // // Now create the BVH for the stop signs
  // // Since the stop signs never move we only need to define this once
  const int64_t nStopSigns = stopSigns.size();
  if (nStopSigns > 0) {
    std::vector<const geometry::AABBInterface*> stopSignObjects;
    stopSignObjects.reserve(nStopSigns);
    for (const auto& obj : stopSigns) {
      stopSignObjects.push_back(
          dynamic_cast<const geometry::AABBInterface*>(obj.get()));
    }
    stop_sign_bvh.InitHierarchy(stopSignObjects);
  }

  // // Now create the BVH for the traffic lights
  // // Since the line segments never move we only need to define this once but
  // // we also need to push back an index into the traffic light states so that
  // // we can
  const int64_t nTrafficLights = trafficLights.size();
  if (nTrafficLights > 0) {
    std::vector<const geometry::AABBInterface*> tlObjects;
    tlObjects.reserve(nTrafficLights);
    for (const auto& obj : trafficLights) {
      tlObjects.push_back(dynamic_cast<geometry::AABBInterface*>(obj.get()));
    }
    tl_bvh_.InitHierarchy(tlObjects);
  }
}

// void Scenario::createVehicle(float posX, float posY, float width, float
// length,
//                              float heading, bool occludes, bool collides,
//                              bool checkForCollisions, float goalPosX,
//                              float goalPosY) {
//   Vehicle* veh = new Vehicle(geometry::Vector2D(posX, posY), width, length,
//                              heading, occludes, collides, checkForCollisions,
//                              geometry::Vector2D(goalPosX, goalPosY));
//   auto ptr = std::shared_ptr<Vehicle>(veh);
//   vehicles.push_back(ptr);
// }

void Scenario::step(float dt) {
  currTime += int(dt / 0.1);  // TODO(ev) hardcoding
  for (auto& object : roadObjects) {
    object->Step(dt);
  }
  for (auto& object : trafficLights) {
    object->set_current_time(currTime);
  }

  // update the vehicle bvh
  const int64_t n = roadObjects.size();
  if (n > 0) {
    std::vector<const geometry::AABBInterface*> objects;
    objects.reserve(n);
    for (const auto& obj : roadObjects) {
      objects.push_back(
          dynamic_cast<const geometry::AABBInterface*>(obj.get()));
    }
    vehicle_bvh_.InitHierarchy(objects);
    // check vehicle-vehicle collisions
    for (auto& obj1 : roadObjects) {
      std::vector<const geometry::AABBInterface*> candidates =
          vehicle_bvh_.IntersectionCandidates(*obj1);
      for (const auto* ptr : candidates) {
        const KineticObject* obj2 = dynamic_cast<const KineticObject*>(ptr);
        if (obj1->id() == obj2->id()) {
          continue;
        }
        if (!obj1->can_be_collided() || !obj2->can_be_collided()) {
          continue;
        }
        if (!obj1->check_collision() && !obj2->check_collision()) {
          continue;
        }
        if (checkForCollision(*obj1, *obj2)) {
          obj1->set_collided(true);
          const_cast<KineticObject*>(obj2)->set_collided(true);
        }
      }
    }
  }

  // check vehicle-lane segment collisions
  for (auto& obj1 : roadObjects) {
    std::vector<const geometry::AABBInterface*> candidates =
        line_segment_bvh_.IntersectionCandidates(*obj1);
    for (const auto* ptr : candidates) {
      const geometry::LineSegment* obj2 =
          dynamic_cast<const geometry::LineSegment*>(ptr);
      if (checkForCollision(*obj1, *obj2)) {
        obj1->set_collided(true);
      }
    }
  }
}

void Scenario::waymo_step() {
  if (currTime < 91) {
    currTime += 1;  // TODO(ev) hardcoding
    for (auto& object : roadObjects) {
      geometry::Vector2D expertPosition =
          expertTrajectories[object->id()][currTime];
      object->set_position(expertPosition);
      object->set_heading(expertHeadings[object->id()][currTime]);
    }
    for (auto& object : trafficLights) {
      object->set_current_time(currTime);
    }

    // initalize the vehicle bvh
    const int64_t n = roadObjects.size();
    if (n > 0) {
      std::vector<const geometry::AABBInterface*> objects;
      objects.reserve(n);
      for (const auto& obj : roadObjects) {
        objects.push_back(
            dynamic_cast<const geometry::AABBInterface*>(obj.get()));
      }
      vehicle_bvh_.InitHierarchy(objects);
      // check vehicle-vehicle collisions
      for (auto& obj1 : roadObjects) {
        std::vector<const geometry::AABBInterface*> candidates =
            vehicle_bvh_.IntersectionCandidates(*obj1);
        for (const auto* ptr : candidates) {
          const KineticObject* obj2 = dynamic_cast<const KineticObject*>(ptr);
          if (obj1->id() == obj2->id()) {
            continue;
          }
          if (!obj1->can_be_collided() || !obj2->can_be_collided()) {
            continue;
          }
          if (!obj1->check_collision() && !obj2->check_collision()) {
            continue;
          }
          if (checkForCollision(*obj1, *obj2)) {
            obj1->set_collided(true);
            const_cast<KineticObject*>(obj2)->set_collided(true);
          }
        }
      }
    }
    // check vehicle-lane segment collisions
    for (auto& obj1 : roadObjects) {
      std::vector<const geometry::AABBInterface*> candidates =
          line_segment_bvh_.IntersectionCandidates(*obj1);
      for (const auto* ptr : candidates) {
        const geometry::LineSegment* obj2 =
            dynamic_cast<const geometry::LineSegment*>(ptr);
        if (checkForCollision(*obj1, *obj2)) {
          obj1->set_collided(true);
        }
      }
    }
  }
}

std::pair<float, geometry::Vector2D> Scenario::getObjectHeadingAndPos(
    KineticObject* sourceObject) {
  float sourceHeading =
      geometry::utils::NormalizeAngle(sourceObject->heading());
  geometry::Vector2D sourcePos = sourceObject->position();
  return std::make_pair(sourceHeading, sourcePos);
}

const geometry::Box* Scenario::getOuterBox(float sourceHeading,
                                           geometry::Vector2D sourcePos,
                                           float halfViewAngle,
                                           float viewDist) {
  const geometry::Box* outerBox;
  float leftAngle =
      sourceHeading +
      geometry::utils::kPi / 4.0f;  // the angle pointing to the top left corner
                                    // of the rectangle enclosing the cone
  float scalingFactorLeft =
      viewDist / std::cos(halfViewAngle);  // how long the vector should be
  geometry::Vector2D topLeft =
      geometry::Vector2D(std::cos(leftAngle), std::sin(leftAngle));
  topLeft *= scalingFactorLeft;
  topLeft += sourcePos;
  float scalingFactorRight =
      viewDist / std::sin((geometry::utils::kPi / 2.0) - halfViewAngle);
  float rightAngle =
      sourceHeading - 3 * geometry::utils::kPi /
                          4.0f;  // the angle pointing to the bottom right hand
                                 // corner of the rectangle enclosing the cone
  geometry::Vector2D bottomRight =
      geometry::Vector2D(std::cos(rightAngle), std::sin(rightAngle));
  bottomRight *= scalingFactorRight;
  bottomRight += sourcePos;
  outerBox = new geometry::Box(topLeft, bottomRight);
  return outerBox;
}

std::vector<float> Scenario::getVisibleObjects(KineticObject* sourceObj,
                                               float viewAngle,
                                               float viewDist) {
  auto [sourceHeading, sourcePos] = getObjectHeadingAndPos(sourceObj);
  float halfViewAngle = viewAngle / 2.0;
  int statePosCounter = 0;
  std::vector<float> state(maxNumVisibleObjects * 4);
  std::fill(state.begin(), state.end(), -100);  // TODO(ev hardcoding)
  const geometry::Box* outerBox =
      getOuterBox(sourceHeading, sourcePos, halfViewAngle, viewDist);
  if (!vehicle_bvh_.Empty()) {
    std::vector<std::tuple<const KineticObject*, float, float>> visibleVehicles;
    std::vector<const geometry::AABBInterface*> roadObjCandidates =
        vehicle_bvh_.IntersectionCandidates(*outerBox);
    for (const auto* ptr : roadObjCandidates) {
      const KineticObject* objPtr = dynamic_cast<const KineticObject*>(ptr);
      if (objPtr->id() == sourceObj->id()) {
        continue;
      }
      geometry::Vector2D otherRelativePos = objPtr->position() - sourcePos;
      float dist = otherRelativePos.Norm();

      if (dist > viewDist) {
        continue;
      }

      float otherRelativeHeading = otherRelativePos.Angle();

      float headingDiff = getSignedAngle(sourceHeading, otherRelativeHeading);

      if (std::abs(headingDiff) <= halfViewAngle) {
        visibleVehicles.push_back(std::make_tuple(objPtr, dist, headingDiff));
      }
    }

    // we want all the vehicles sorted by distance to the agent
    std::sort(visibleVehicles.begin(), visibleVehicles.end(),
              [](auto a, auto b) { return std::get<1>(a) < std::get<1>(b); });

    int nVeh = visibleVehicles.size();
    for (int k = 0; k < std::min(nVeh, maxNumVisibleObjects); ++k) {
      auto vehData = visibleVehicles[k];
      state[statePosCounter++] = std::get<1>(vehData);
      state[statePosCounter++] = std::get<2>(vehData);
      state[statePosCounter++] = std::get<0>(vehData)->speed();
      state[statePosCounter++] = std::get<0>(vehData)->length();
    }
    // increment the state counter appropriately if we have fewer than the
    // maximum number of vehicles
    if (nVeh < maxNumVisibleObjects) {
      statePosCounter += (maxNumVisibleObjects - nVeh) * 4;
    }
  } else {
    statePosCounter += maxNumVisibleObjects * 4;
  }
  return state;
}

std::vector<float> Scenario::getVisibleRoadPoints(KineticObject* sourceObj,
                                                  float viewAngle,
                                                  float viewDist) {
  // auto [sourceHeading, sourcePos] = getObjectHeadingAndPos(sourceObj);
  // float halfViewAngle = viewAngle / 2.0;
  // int statePosCounter = 0;
  // std::vector<float> state(maxNumVisibleRoadPoints * 3);
  // std::fill(state.begin(), state.end(), -100);  // TODO(ev hardcoding)
  // const geometry::Box* outerBox =
  //     getOuterBox(sourceHeading, sourcePos, halfViewAngle, viewDist);
  // if (!road_point_bvh.Empty()) {
  //   std::vector<std::tuple<const RoadPoint*, float, float>> visibleRoadPoints;
  //   std::vector<const geometry::AABBInterface*> roadPointCandidates =
  //       road_point_bvh.IntersectionCandidates(*outerBox);
  //   for (const auto* ptr : roadPointCandidates) {
  //     const RoadPoint* objPtr = dynamic_cast<const RoadPoint*>(ptr);
  //     geometry::Vector2D otherRelativePos = objPtr->position() - sourcePos;
  //     float dist = otherRelativePos.Norm();
  //
  //     if (dist > viewDist) {
  //       continue;
  //     }
  //
  //     float otherRelativeHeading = otherRelativePos.Angle();
  //     float headingDiff = getSignedAngle(sourceHeading, otherRelativeHeading);
  //
  //     if (std::abs(headingDiff) <= halfViewAngle) {
  //       visibleRoadPoints.push_back(std::make_tuple(objPtr, dist, headingDiff));
  //     }
  //   }
  //   // we want all the road points sorted by distance to the agent
  //   std::sort(visibleRoadPoints.begin(), visibleRoadPoints.end(),
  //             [](auto a, auto b) { return std::get<1>(a) < std::get<1>(b); });
  //   int nRoadPoints = visibleRoadPoints.size();
  //   for (int k = 0; k < std::min(nRoadPoints, maxNumVisibleRoadPoints); ++k) {
  //     auto pointData = visibleRoadPoints[k];
  //     state[statePosCounter++] = std::get<1>(pointData);
  //     state[statePosCounter++] = std::get<2>(pointData);
  //     state[statePosCounter++] = float(std::get<0>(pointData)->road_type());
  //   }
  //   // increment the state counter appropriately if we have fewer than the
  //   // maximum number of visible road points
  //   if (nRoadPoints < maxNumVisibleRoadPoints) {
  //     statePosCounter += (maxNumVisibleRoadPoints - nRoadPoints) * 3;
  //   }
  // } else {
  //   statePosCounter += maxNumVisibleRoadPoints * 3;
  // }
  // return state;
}

std::vector<float> Scenario::getVisibleStopSigns(KineticObject* sourceObj,
                                                 float viewAngle,
                                                 float viewDist) {
  auto [sourceHeading, sourcePos] = getObjectHeadingAndPos(sourceObj);
  float halfViewAngle = viewAngle / 2.0;
  int statePosCounter = 0;
  std::vector<float> state(maxNumVisibleStopSigns * 2);
  std::fill(state.begin(), state.end(), -100);  // TODO(ev hardcoding)
  const geometry::Box* outerBox =
      getOuterBox(sourceHeading, sourcePos, halfViewAngle, viewDist);
  if (!stop_sign_bvh.Empty()) {
    std::vector<std::tuple<float, float>> visibleStopSigns;
    std::vector<const geometry::AABBInterface*> stopSignCandidates =
        stop_sign_bvh.IntersectionCandidates(*outerBox);
    for (const auto* ptr : stopSignCandidates) {
      const geometry::Box* objPtr = dynamic_cast<const geometry::Box*>(ptr);
      geometry::Vector2D otherRelativePos =
          ((objPtr->Endpoint0() + objPtr->Endpoint1()) / 2.0) - sourcePos;
      float dist = otherRelativePos.Norm();

      if (dist > viewDist) {
        continue;
      }

      float otherRelativeHeading = otherRelativePos.Angle();
      float headingDiff = getSignedAngle(sourceHeading, otherRelativeHeading);

      if (std::abs(headingDiff) <= halfViewAngle) {
        visibleStopSigns.push_back(std::make_tuple(dist, headingDiff));
      }
    }

    // we want all the stop signs sorted by distance to the agent
    std::sort(visibleStopSigns.begin(), visibleStopSigns.end(),
              [](auto a, auto b) { return std::get<1>(a) < std::get<1>(b); });

    int nStopSigns = visibleStopSigns.size();
    for (size_t k = 0; k < std::min(nStopSigns, maxNumVisibleStopSigns); ++k) {
      auto pointData = visibleStopSigns[k];
      state[statePosCounter++] = std::get<0>(pointData);
      state[statePosCounter++] = std::get<1>(pointData);
    }
    // increment the state counter appropriately if we have fewer than the
    // maximum number of visible stop signs
    if (nStopSigns < maxNumVisibleStopSigns) {
      statePosCounter += (maxNumVisibleStopSigns - nStopSigns) * 2;
    }
  } else {
    statePosCounter += maxNumVisibleStopSigns * 2;
  }
  return state;
}

std::vector<float> Scenario::getVisibleTrafficLights(KineticObject* sourceObj,
                                                     float viewAngle,
                                                     float viewDist) {
  auto [sourceHeading, sourcePos] = getObjectHeadingAndPos(sourceObj);
  float halfViewAngle = viewAngle / 2.0;
  int statePosCounter = 0;
  std::vector<float> state(maxNumVisibleTLSigns * 3);
  std::fill(state.begin(), state.end(), -100);  // TODO(ev hardcoding)
  // const geometry::Box* outerBox =
  //     getOuterBox(sourceHeading, sourcePos, halfViewAngle, viewDist);
  // if (!tl_bvh_.Empty()) {
  //   std::vector<std::tuple<float, float, int>> visibleTrafficLights;
  //   std::vector<const geometry::AABBInterface*> tlCandidates =
  //       tl_bvh_.IntersectionCandidates(*outerBox);
  //   for (const auto* ptr : tlCandidates) {
  //     const TrafficLightBox* objPtr = dynamic_cast<const
  //     TrafficLightBox*>(ptr); geometry::Vector2D otherRelativePos =
  //     objPtr->position - sourcePos; float dist = otherRelativePos.Norm();
  //
  //     if (dist > viewDist) {
  //       continue;
  //     }
  //
  //     float otherRelativeHeading = otherRelativePos.Angle();
  //
  //     float headingDiff = getSignedAngle(sourceHeading,
  //     otherRelativeHeading);
  //
  //     if (std::abs(headingDiff) <= halfViewAngle) {
  //       visibleTrafficLights.push_back(
  //           std::make_tuple(dist, headingDiff,
  //                           trafficLights[objPtr->tlIndex]->getLightState()));
  //     }
  //   }
  //
  //   // we want all the traffic lights sorted by distance to the agent
  //   std::sort(visibleTrafficLights.begin(), visibleTrafficLights.end(),
  //             [](auto a, auto b) { return std::get<1>(a) < std::get<1>(b);
  //             });
  //
  //   int nTrafficLights = visibleTrafficLights.size();
  //   for (size_t k = 0; k < std::min(nTrafficLights, maxNumVisibleTLSigns);
  //        ++k) {
  //     auto pointData = visibleTrafficLights[k];
  //     state[statePosCounter++] = std::get<0>(pointData);
  //     state[statePosCounter++] = std::get<1>(pointData);
  //     state[statePosCounter++] = std::get<2>(pointData);
  //   }
  //   // increment the state counter appropriately if we have fewer than the
  //   // maximum number of visible traffic lights
  //   if (nTrafficLights < maxNumVisibleTLSigns) {
  //     statePosCounter += (maxNumVisibleTLSigns - nTrafficLights) * 3;
  //   }
  // } else {
  //   statePosCounter += maxNumVisibleTLSigns * 3;
  // }

  return state;
}

std::vector<float> Scenario::getVisibleState(KineticObject* sourceObj,
                                             float viewAngle, float viewDist) {
  // Vehicle state is: dist, heading-diff, speed, length
  // TL state is: dist, heading-diff, current light color as an int
  // Road point state is: dist, heading-diff, type of road point as an int i.e.
  // roadEdge, laneEdge, etc. Stop sign state is: dist, heading-diff
  const int64_t n = maxNumVisibleObjects * 4 + maxNumVisibleTLSigns * 3 +
                    maxNumVisibleRoadPoints * 3 + 2 * maxNumVisibleStopSigns;
  std::vector<float> state;
  state.reserve(n);
  std::vector<float> vehicleState =
      getVisibleObjects(sourceObj, viewAngle, viewDist);
  std::vector<float> roadState =
      getVisibleRoadPoints(sourceObj, viewAngle, viewDist);
  std::vector<float> stopSignState =
      getVisibleStopSigns(sourceObj, viewAngle, viewDist);
  std::vector<float> tlState =
      getVisibleTrafficLights(sourceObj, viewAngle, viewDist);
  state.insert(state.end(), vehicleState.begin(), vehicleState.end());
  state.insert(state.end(), roadState.begin(), roadState.end());
  state.insert(state.end(), stopSignState.begin(), stopSignState.end());
  state.insert(state.end(), tlState.begin(), tlState.end());
  for (float i : state) {
    std::cout << i << std::endl;
  }
  return state;
}

std::vector<float> Scenario::getEgoState(KineticObject* obj) {
  std::vector<float> state(4);

  state[0] = obj->speed();

  float sourceHeading = geometry::utils::NormalizeAngle(obj->heading());

  geometry::Vector2D sourcePos = obj->position();
  geometry::Vector2D goalPos = obj->destination();

  geometry::Vector2D otherRelativePos = goalPos - sourcePos;
  float dist = otherRelativePos.Norm();

  float otherRelativeHeading = otherRelativePos.Angle();
  float headingDiff = getSignedAngle(sourceHeading, otherRelativeHeading);

  state[1] = dist;
  state[2] = headingDiff;
  state[3] = obj->length();

  return state;
}

bool Scenario::checkForCollision(const Object& object1,
                                 const Object& object2) const {
  // note: right now objects are rectangles but this code works for any pair of
  // convex polygons

  // first check for circles collision
  const float dist = geometry::Distance(object1.position(), object2.position());
  const float min_dist = object1.Radius() + object2.Radius();
  if (dist > min_dist) {
    return false;
  }
  const geometry::ConvexPolygon polygon1 = object1.BoundingPolygon();
  const geometry::ConvexPolygon polygon2 = object2.BoundingPolygon();
  return polygon1.Intersects(polygon2);
}

bool Scenario::checkForCollision(const Object& object,
                                 const geometry::LineSegment& segment) const {
  return geometry::Intersects(segment, object.BoundingPolygon());
}

// TODO(ev) make smoother, also maybe return something named so that
// it's clear what's accel and what's steeringAngle
std::vector<float> Scenario::getExpertAction(int objID, int timeIdx) {
  // we want to return accel, steering angle
  // so first we get the accel
  geometry::Vector2D accel_vec =
      (expertSpeeds[objID][timeIdx + 1] - expertSpeeds[objID][timeIdx - 1]) /
      0.2;
  float accel = accel_vec.Norm();
  float speed = expertSpeeds[objID][timeIdx].Norm();
  float dHeading = (expertHeadings[objID][timeIdx + 1] -
                    expertHeadings[objID][timeIdx - 1]) /
                   0.2;
  float steeringAngle;
  if (speed > 0.0) {
    steeringAngle = asin(dHeading / speed * lengths[objID]);
  } else {
    steeringAngle = 0.0;
  }
  std::vector<float> expertAction = {accel, steeringAngle};
  return expertAction;
};

bool Scenario::hasExpertAction(int objID, unsigned int timeIdx) {
  // The user requested too large a point or a point that
  // can't be used for a second order expansion
  if (timeIdx > expertValid[objID].size() - 1 || timeIdx < 1) {
    return false;
  } else if (!expertValid[objID][timeIdx - 1] ||
             !expertValid[objID][timeIdx + 1]) {
    return false;
  } else {
    return true;
  }
}

std::vector<bool> Scenario::getValidExpertStates(int objID) {
  return expertValid[objID];
}

void Scenario::draw(sf::RenderTarget& target, sf::RenderStates states) const {
  // for (const auto& road : roads) {
  //   target.draw(*road, states);
  // }
  for (const auto& object : roadLines) {
    target.draw(*object, states);
  }
  for (const auto& object : trafficLights) {
    target.draw(*object, states);
  }
  for (const auto& object : stopSigns) {
    target.draw(*object, states);
  }
  for (const auto& object : roadObjects) {
    target.draw(*object, states);
    // draw goal destination
    float radius = 2;
    sf::CircleShape ptShape(radius);
    ptShape.setOrigin(radius, radius);
    ptShape.setFillColor(object->color());
    ptShape.setPosition(utils::ToVector2f(object->destination()));
    target.draw(ptShape, states);
  }
}

std::vector<std::shared_ptr<Vehicle>> Scenario::getVehicles() {
  return vehicles;
}
std::vector<std::shared_ptr<Pedestrian>> Scenario::getPedestrians() {
  return pedestrians;
}
std::vector<std::shared_ptr<Cyclist>> Scenario::getCyclists() {
  return cyclists;
}
std::vector<std::shared_ptr<KineticObject>> Scenario::getRoadObjects() {
  return roadObjects;
}
std::vector<std::shared_ptr<RoadLine>> Scenario::getRoadLines() {
  return roadLines;
}

float Scenario::getSignedAngle(float sourceAngle, float targetAngle) const {
  return geometry::utils::AngleSub(targetAngle, sourceAngle);
}

void Scenario::removeVehicle(Vehicle* object) {
  for (auto it = vehicles.begin(); it != vehicles.end();) {
    if ((*it).get() == object) {
      it = vehicles.erase(it);
    } else {
      it++;
    }
  }

  for (auto it = roadObjects.begin(); it != roadObjects.end();) {
    if ((*it).get() == object) {
      it = roadObjects.erase(it);
    } else {
      it++;
    }
  }
}

sf::FloatRect Scenario::getRoadNetworkBoundaries() const {
  return roadNetworkBounds;
}

ImageMatrix Scenario::getCone(KineticObject* object, float viewAngle,
                              float viewDist, float headTilt,
                              bool obscuredView) {  // args in radians
  float circleRadius = viewDist;
  float renderedCircleRadius = 300.0f;

  if (object->cone_texture() == nullptr) {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 1;

    // object->coneTexture = new sf::RenderTexture();
    // object->coneTexture->create(2.0f * renderedCircleRadius,
    //                             2.0f * renderedCircleRadius, settings);

    // Potential memory leak?
    object->set_cone_texture(new sf::RenderTexture());
    object->cone_texture()->create(2.0f * renderedCircleRadius,
                                   2.0f * renderedCircleRadius, settings);
  }
  sf::RenderTexture* texture = object->cone_texture();

  sf::Transform renderTransform;
  renderTransform.scale(1, -1);  // horizontal flip

  geometry::Vector2D center = object->position();

  texture->clear(sf::Color(50, 50, 50));
  sf::View view(utils::ToVector2f(center, /*flip_y=*/true),
                sf::Vector2f(2.0f * circleRadius, 2.0f * circleRadius));
  view.rotate(-geometry::utils::Degrees(object->heading()) + 90.0f);
  texture->setView(view);

  texture->draw(
      *this,
      renderTransform);  // todo optimize with objects in range only (quadtree?)

  texture->setView(
      sf::View(sf::Vector2f(0.0f, 0.0f), sf::Vector2f(texture->getSize())));

  // draw circle
  float r = renderedCircleRadius;
  float diag = std::sqrt(2 * r * r);

  for (int quadrant = 0; quadrant < 4; ++quadrant) {
    std::vector<sf::Vertex> outerCircle;  // todo precompute just once

    float angleShift = quadrant * kPi / 2.0f;

    geometry::Vector2D corner =
        geometry::PolarToVector2D(diag, kPi / 4.0f + angleShift);
    outerCircle.push_back(
        sf::Vertex(utils::ToVector2f(corner), sf::Color::Black));

    int nPoints = 20;
    for (int i = 0; i < nPoints; ++i) {
      float angle = angleShift + i * (kPi / 2.0f) / (nPoints - 1);

      geometry::Vector2D pt = geometry::PolarToVector2D(r, angle);
      outerCircle.push_back(
          sf::Vertex(utils::ToVector2f(pt), sf::Color::Black));
    }

    texture->draw(&outerCircle[0], outerCircle.size(),
                  sf::TriangleFan);  //, renderTransform);
  }

  renderTransform.rotate(-geometry::utils::Degrees(object->heading()) + 90.0f);

  // TODO(ev) do this for road objects too
  // draw obstructions
  if (obscuredView == true) {
    std::vector<std::shared_ptr<Vehicle>> roadObjects =
        getVehicles();  // todo optimize with objects in range only (quadtree?)

    for (const auto& obj : roadObjects) {
      if (obj.get() != object && obj->can_block_sight()) {
        const auto lines = obj->BoundingPolygon().Edges();
        // auto lines = obj->getLines();
        for (const auto& line1 : lines) {
          const geometry::Vector2D& pt1 = line1.Endpoint0();
          const geometry::Vector2D& pt2 = line1.Endpoint1();
          int nIntersections = 0;
          for (const auto& line2 : lines) {
            const geometry::Vector2D& pt3 = line2.Endpoint0();
            const geometry::Vector2D& pt4 = line2.Endpoint1();
            if (pt1 != pt3 && pt1 != pt4 &&
                geometry::LineSegment(pt1, center).Intersects(line2)) {
              nIntersections++;
              break;
            }
          }
          for (const auto& line2 : lines) {
            const geometry::Vector2D& pt3 = line2.Endpoint0();
            const geometry::Vector2D& pt4 = line2.Endpoint1();
            if (pt2 != pt3 && pt2 != pt4 &&
                geometry::LineSegment(pt2, center).Intersects(line2)) {
              nIntersections++;
              break;
            }
          }

          if (nIntersections >= 1) {
            sf::ConvexShape hiddenArea;

            float angle1 = (pt1 - center).Angle();
            float angle2 = (pt2 - center).Angle();
            while (angle2 > angle1) angle2 -= 2.0f * geometry::utils::kPi;

            int nPoints = 80;  // todo function of angle
            hiddenArea.setPointCount(nPoints + 2);

            float ratio = renderedCircleRadius / circleRadius;
            hiddenArea.setPoint(0, utils::ToVector2f((pt1 - center) * ratio));
            for (int i = 0; i < nPoints; ++i) {
              float angle = angle1 + i * (angle2 - angle1) / (nPoints - 1);
              geometry::Vector2D pt = geometry::PolarToVector2D(r, angle);
              hiddenArea.setPoint(1 + i, utils::ToVector2f(pt));
            }
            hiddenArea.setPoint(nPoints + 1,
                                utils::ToVector2f((pt2 - center) * ratio));

            hiddenArea.setFillColor(sf::Color::Black);

            texture->draw(hiddenArea, renderTransform);
          }
        }
      }
    }
  }

  // TODO(ev) this represents traffic lights by drawing the
  // light state onto the lane. It should never be obscured.
  // Once we figure out where the traffic light actually is,
  // we can remove this
  sf::Transform renderTransform2;
  renderTransform2.scale(1, -1);

  texture->setView(view);
  for (const auto& object : trafficLights) {
    texture->draw(*object, renderTransform2);
  }

  texture->display();

  // draw cone
  texture->setView(
      sf::View(sf::Vector2f(0.0f, 0.0f), sf::Vector2f(texture->getSize())));

  renderTransform.rotate(geometry::utils::Degrees(object->heading()) - 90.0f);
  if (viewAngle < 2.0f * kPi) {
    std::vector<sf::Vertex> innerCircle;  // todo precompute just once

    innerCircle.push_back(
        sf::Vertex(sf::Vector2f(0.0f, 0.0f), sf::Color::Black));
    float startAngle = kPi / 2.0f + headTilt + viewAngle / 2.0f;
    float endAngle = kPi / 2.0f + headTilt + 2.0f * kPi - viewAngle / 2.0f;

    int nPoints = 80;  // todo function of angle
    for (int i = 0; i < nPoints; ++i) {
      float angle = startAngle + i * (endAngle - startAngle) / (nPoints - 1);
      geometry::Vector2D pt = geometry::PolarToVector2D(r, angle);
      innerCircle.push_back(
          sf::Vertex(utils::ToVector2f(pt), sf::Color::Black));
    }

    texture->draw(&innerCircle[0], innerCircle.size(), sf::TriangleFan,
                  renderTransform);
  }
  texture->display();

  sf::Image img = texture->getTexture().copyToImage();
  unsigned char* pixelsArr = (unsigned char*)img.getPixelsPtr();

  return ImageMatrix(pixelsArr, renderedCircleRadius * 2,
                     renderedCircleRadius * 2, 4);
}

ImageMatrix Scenario::getImage(KineticObject* object, bool renderGoals) {
  int squareSide = 600;

  if (imageTexture == nullptr) {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 4;
    imageTexture = new sf::RenderTexture();
    imageTexture->create(squareSide, squareSide, settings);
  }
  sf::RenderTexture* texture = imageTexture;

  sf::Transform renderTransform;
  renderTransform.scale(1, -1);  // horizontal flip

  texture->clear(sf::Color(50, 50, 50));

  // same as in Simulation.cpp
  float padding = 0.0f;
  sf::FloatRect scenarioBounds = getRoadNetworkBoundaries();
  scenarioBounds.top = -scenarioBounds.top - scenarioBounds.height;
  scenarioBounds.top -= padding;
  scenarioBounds.left -= padding;
  scenarioBounds.width += 2 * padding;
  scenarioBounds.height += 2 * padding;
  sf::Vector2f center =
      sf::Vector2f(scenarioBounds.left + scenarioBounds.width / 2.0f,
                   scenarioBounds.top + scenarioBounds.height / 2.0f);
  sf::Vector2f size = sf::Vector2f(squareSide, squareSide) *
                      std::max(scenarioBounds.width / squareSide,
                               scenarioBounds.height / squareSide);
  sf::View view(center, size);
  if (object != nullptr) {
    view.rotate(-geometry::utils::Degrees(object->heading()) + 90.0f);
  }

  texture->setView(view);

  // for (const auto& road : roads) {
  //   texture->draw(*road, renderTransform);
  // }
  if (object == nullptr) {
    for (const auto& obj : vehicles) {
      texture->draw(*obj, renderTransform);
      if (renderGoals) {
        // draw goal destination
        float radius = 2;
        sf::CircleShape ptShape(radius);
        ptShape.setOrigin(radius, radius);
        ptShape.setFillColor(obj->color());
        ptShape.setPosition(utils::ToVector2f(obj->position()));
        texture->draw(ptShape, renderTransform);
      }
    }
  } else {
    texture->draw(*object, renderTransform);

    if (renderGoals) {
      // draw goal destination
      float radius = 2;
      sf::CircleShape ptShape(radius);
      ptShape.setOrigin(radius, radius);
      ptShape.setFillColor(object->color());
      ptShape.setPosition(utils::ToVector2f(object->position()));
      texture->draw(ptShape, renderTransform);
    }
  }
  for (const auto& obj : roadLines) {
    texture->draw(*obj, renderTransform);
  }
  for (const auto& obj : trafficLights) {
    texture->draw(*obj, renderTransform);
  }
  for (const auto& obj : stopSigns) {
    texture->draw(*obj, renderTransform);
  }
  // render texture and return
  texture->display();

  sf::Image img = texture->getTexture().copyToImage();
  unsigned char* pixelsArr = (unsigned char*)img.getPixelsPtr();

  return ImageMatrix(pixelsArr, squareSide, squareSide, 4);
}

}  // namespace nocturne
