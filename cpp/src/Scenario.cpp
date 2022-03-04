#include "Scenario.hpp"

#include <algorithm>
#include <limits>

#include "geometry/aabb_interface.h"
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
    float heading =
        geometry::utils::Radians(static_cast<float>(obj["heading"][currTime]));

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
      float heading = obj["heading"][i];
      localHeadingVec.push_back(heading * float(geometry::utils::kPi / 180.0));
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
        Vehicle* veh = new Vehicle(
            pos, width, length, heading, occludes, collides, checkForCollisions,
            goalPos, IDCounter, localExpertSpeeds[currTime].Norm());
        auto ptr = std::shared_ptr<Vehicle>(veh);
        vehicles.push_back(ptr);
        roadObjects.push_back(ptr);
        if (didObjectMove) {
          objectsThatMoved.push_back(ptr);
        }
      } else if (type == "pedestrian" && useNonVehicles) {
        Pedestrian* ped = new Pedestrian(
            pos, width, length, heading, occludes, collides, checkForCollisions,
            goalPos, IDCounter, localExpertSpeeds[currTime].Norm());
        auto ptr = std::shared_ptr<Pedestrian>(ped);
        pedestrians.push_back(ptr);
        roadObjects.push_back(ptr);
        if (didObjectMove) {
          objectsThatMoved.push_back(ptr);
        }
      } else if (type == "cyclist" && useNonVehicles) {
        Cyclist* cyclist = new Cyclist(
            pos, width, length, heading, occludes, collides, checkForCollisions,
            goalPos, IDCounter, localExpertSpeeds[currTime].Norm());
        auto ptr = std::shared_ptr<Cyclist>(cyclist);
        cyclists.push_back(ptr);
        roadObjects.push_back(ptr);
        if (didObjectMove) {
          objectsThatMoved.push_back(ptr);
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
    bool checkForCollisions = false;
    RoadType road_type;
    // TODO(ev) this is not good
    if (type == "lane") {
      road_type = RoadType::lane;
    } else if (type == "road_line") {
      road_type = RoadType::road_line;
    } else if (type == "road_edge") {
      road_type = RoadType::road_edge;
      checkForCollisions = true;
    } else if (type == "stop_sign") {
      road_type = RoadType::stop_sign;
    } else if (type == "crosswalk") {
      road_type = RoadType::crosswalk;
    } else if (type == "speed_bump") {
      road_type = RoadType::speed_bump;
    } else {
      road_type = RoadType::none;
    }
    // we have to handle stop signs differently from other lane types
    if (road_type == RoadType::stop_sign) {
      stopSigns.push_back(geometry::Vector2D(road["geometry"][0]["x"],
                                             road["geometry"][0]["y"]));
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
          geometry::LineSegment* lineSegment =
              new geometry::LineSegment(startPoint, endPoint);
          lineSegments.push_back(
              std::shared_ptr<geometry::LineSegment>(lineSegment));
        }
      }
    }
    // Now construct the entire roadline object which is what will be used for
    // drawing
    int road_size = road["geometry"].size();
    laneGeometry.emplace_back(road["geometry"][road_size - 1]["x"],
                              road["geometry"][road_size - 1]["y"]);
    RoadLine* roadLine =
        new RoadLine(laneGeometry, road_type, checkForCollisions);
    roadLines.push_back(std::shared_ptr<RoadLine>(roadLine));
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
    std::vector<int> validTimes;
    std::vector<LightState> lightStates;

    for (size_t i = 0; i < tl["state"].size(); i++) {
      // TODO(ev) do this more compactly
      LightState lightState;
      if (tl["state"][i] == "unknown") {
        lightState = LightState::unknown;
      } else if (tl["state"][i] == "arrow_stop") {
        lightState = LightState::arrow_stop;
      } else if (tl["state"][i] == "arrow_caution") {
        lightState = LightState::arrow_caution;
      } else if (tl["state"][i] == "arrow_go") {
        lightState = LightState::arrow_go;
      } else if (tl["state"][i] == "stop") {
        lightState = LightState::stop;
      } else if (tl["state"][i] == "caution") {
        lightState = LightState::caution;
      } else if (tl["state"][i] == "go") {
        lightState = LightState::go;
      } else if (tl["state"][i] == "flashing_stop") {
        lightState = LightState::flashing_stop;
      } else if (tl["state"][i] == "flashing_caution") {
        lightState = LightState::flashing_caution;
      }
      lightStates.push_back(lightState);
      validTimes.push_back(int(tl["time_index"][i]));
    }
    TrafficLight* t_light =
        new TrafficLight(x_pos, y_pos, lightStates, currTime, validTimes);
    auto ptr = std::shared_ptr<TrafficLight>(t_light);
    trafficLights.push_back(ptr);
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
  const int64_t nPointsPerLine = roadLines[0]->getRoadPoints().size();
  if ((nLines * nPointsPerLine) > 0) {
    std::vector<const geometry::AABBInterface*> roadPointObjects;
    roadPointObjects.reserve(nLines * nPointsPerLine);
    for (const auto& roadLine : roadLines) {
      for (const auto& roadPoint : roadLine->getRoadPoints()) {
        roadPointObjects.push_back(
            dynamic_cast<const geometry::AABBInterface*>(roadPoint.get()));
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
      // create a box to store
      geometry::Vector2D rightElement = obj + 2;
      geometry::Vector2D leftElement = obj - 2;
      geometry::Box* box = new geometry::Box(leftElement, rightElement);
      stopSignObjects.push_back(
          dynamic_cast<const geometry::AABBInterface*>(box));
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
    int i = 0;
    for (const auto& obj : trafficLights) {
      TrafficLightBox* tlBox = new TrafficLightBox(obj->getPosition(), i);
      tlObjects.push_back(dynamic_cast<const geometry::AABBInterface*>(tlBox));
      i++;
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
    object->step(dt);
  }
  for (auto& object : trafficLights) {
    object->updateTime(currTime);
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
        const Object* obj2 = dynamic_cast<const Object*>(ptr);
        if (obj1->getID() == obj2->getID()) {
          continue;
        }
        if (!obj1->checkForCollisions && !obj2->checkForCollisions) {
          continue;
        }
        if (!obj1->collides || !obj2->collides) {
          continue;
        }
        if (checkForCollision(obj1.get(), obj2)) {
          obj1->setCollided(true);
          const_cast<Object*>(obj2)->setCollided(true);
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
      if (checkForCollision(obj1.get(), obj2)) {
        obj1->setCollided(true);
      }
    }
  }
}

void Scenario::waymo_step() {
  if (currTime < 91) {
    currTime += 1;  // TODO(ev) hardcoding
    for (auto& object : roadObjects) {
      geometry::Vector2D expertPosition =
          expertTrajectories[int(object->getID())][currTime];
      object->setPosition(expertPosition.x(), expertPosition.y());
      object->setHeading(expertHeadings[int(object->getID())][currTime]);
    }
    for (auto& object : trafficLights) {
      object->updateTime(currTime);
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
          const Object* obj2 = dynamic_cast<const Object*>(ptr);
          if (obj1->getID() == obj2->getID()) {
            continue;
          }
          if (!obj1->checkForCollisions && !obj2->checkForCollisions) {
            continue;
          }
          if (!obj1->collides || !obj2->collides) {
            continue;
          }
          if (checkForCollision(obj1.get(), obj2)) {
            obj1->setCollided(true);
            const_cast<Object*>(obj2)->setCollided(true);
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
        if (checkForCollision(obj1.get(), obj2)) {
          obj1->setCollided(true);
        }
      }
    }
  }
}

std::pair<float, geometry::Vector2D> Scenario::getObjectHeadingAndPos(
    Object* sourceObject) {
  float sourceHeading = sourceObject->getHeading();
  while (sourceHeading > geometry::utils::kPi)
    sourceHeading -= 2.0f * geometry::utils::kPi;
  while (sourceHeading < -geometry::utils::kPi)
    sourceHeading += 2.0f * geometry::utils::kPi;

  geometry::Vector2D sourcePos = sourceObject->getPosition();
  return std::pair<float, geometry::Vector2D>(sourceHeading, sourcePos);
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

std::vector<float> Scenario::getVisibleObjects(Object* sourceObj,
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
    std::vector<std::tuple<const Object*, float, float>> visibleVehicles;
    std::vector<const geometry::AABBInterface*> roadObjCandidates =
        vehicle_bvh_.IntersectionCandidates(*outerBox);
    for (const auto* ptr : roadObjCandidates) {
      const Object* objPtr = dynamic_cast<const Object*>(ptr);
      if (objPtr->getID() == sourceObj->getID()) {
        continue;
      }
      geometry::Vector2D otherRelativePos = objPtr->getPosition() - sourcePos;
      float dist = otherRelativePos.Norm();

      if (dist > viewDist) {
        continue;
      }

      float otherRelativeHeading = otherRelativePos.Angle();

      float headingDiff = otherRelativeHeading - sourceHeading;
      if (headingDiff > geometry::utils::kPi) {
        headingDiff -= 2.0f * geometry::utils::kPi;
      }
      if (headingDiff < -geometry::utils::kPi) {
        headingDiff += 2.0f * geometry::utils::kPi;
      }

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
      state[statePosCounter++] = std::get<0>(vehData)->getSpeed();
      state[statePosCounter++] = std::get<0>(vehData)->getLength();
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
std::vector<float> Scenario::getVisibleRoadPoints(Object* sourceObj,
                                                  float viewAngle,
                                                  float viewDist) {
  auto [sourceHeading, sourcePos] = getObjectHeadingAndPos(sourceObj);
  float halfViewAngle = viewAngle / 2.0;
  int statePosCounter = 0;
  std::vector<float> state(maxNumVisibleRoadPoints * 3);
  std::fill(state.begin(), state.end(), -100);  // TODO(ev hardcoding)
  const geometry::Box* outerBox =
      getOuterBox(sourceHeading, sourcePos, halfViewAngle, viewDist);
  if (!road_point_bvh.Empty()) {
    std::vector<std::tuple<const RoadPoint*, float, float>> visibleRoadPoints;
    std::vector<const geometry::AABBInterface*> roadPointCandidates =
        road_point_bvh.IntersectionCandidates(*outerBox);
    for (const auto* ptr : roadPointCandidates) {
      const RoadPoint* objPtr = dynamic_cast<const RoadPoint*>(ptr);
      geometry::Vector2D otherRelativePos = objPtr->position - sourcePos;
      float dist = otherRelativePos.Norm();

      if (dist > viewDist) {
        continue;
      }

      float otherRelativeHeading = otherRelativePos.Angle();

      float headingDiff = otherRelativeHeading - sourceHeading;
      if (headingDiff > geometry::utils::kPi) {
        headingDiff -= 2.0f * geometry::utils::kPi;
      }
      if (headingDiff < -geometry::utils::kPi) {
        headingDiff += 2.0f * geometry::utils::kPi;
      }

      if (std::abs(headingDiff) <= halfViewAngle) {
        visibleRoadPoints.push_back(std::make_tuple(objPtr, dist, headingDiff));
      }
    }
    // we want all the road points sorted by distance to the agent
    std::sort(visibleRoadPoints.begin(), visibleRoadPoints.end(),
              [](auto a, auto b) { return std::get<1>(a) < std::get<1>(b); });
    int nRoadPoints = visibleRoadPoints.size();
    for (int k = 0; k < std::min(nRoadPoints, maxNumVisibleRoadPoints); ++k) {
      auto pointData = visibleRoadPoints[k];
      state[statePosCounter++] = std::get<1>(pointData);
      state[statePosCounter++] = std::get<2>(pointData);
      state[statePosCounter++] = float(std::get<0>(pointData)->type);
    }
    // increment the state counter appropriately if we have fewer than the
    // maximum number of visible road points
    if (nRoadPoints < maxNumVisibleRoadPoints) {
      statePosCounter += (maxNumVisibleRoadPoints - nRoadPoints) * 3;
    }
  } else {
    statePosCounter += maxNumVisibleRoadPoints * 3;
  }
  return state;
}

std::vector<float> Scenario::getVisibleStopSigns(Object* sourceObj,
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

      float headingDiff = otherRelativeHeading - sourceHeading;
      if (headingDiff > geometry::utils::kPi) {
        headingDiff -= 2.0f * geometry::utils::kPi;
      }
      if (headingDiff < -geometry::utils::kPi) {
        headingDiff += 2.0f * geometry::utils::kPi;
      }

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

std::vector<float> Scenario::getVisibleTrafficLights(Object* sourceObj,
                                                     float viewAngle,
                                                     float viewDist) {
  auto [sourceHeading, sourcePos] = getObjectHeadingAndPos(sourceObj);
  float halfViewAngle = viewAngle / 2.0;
  int statePosCounter = 0;
  std::vector<float> state(maxNumVisibleTLSigns * 3);
  std::fill(state.begin(), state.end(), -100);  // TODO(ev hardcoding)
  const geometry::Box* outerBox =
      getOuterBox(sourceHeading, sourcePos, halfViewAngle, viewDist);
  if (!tl_bvh_.Empty()) {
    std::vector<std::tuple<float, float, int>> visibleTrafficLights;
    std::vector<const geometry::AABBInterface*> tlCandidates =
        tl_bvh_.IntersectionCandidates(*outerBox);
    for (const auto* ptr : tlCandidates) {
      const TrafficLightBox* objPtr = dynamic_cast<const TrafficLightBox*>(ptr);
      geometry::Vector2D otherRelativePos = objPtr->position - sourcePos;
      float dist = otherRelativePos.Norm();

      if (dist > viewDist) {
        continue;
      }

      float otherRelativeHeading = otherRelativePos.Angle();

      float headingDiff = otherRelativeHeading - sourceHeading;
      if (headingDiff > geometry::utils::kPi) {
        headingDiff -= 2.0f * geometry::utils::kPi;
      }
      if (headingDiff < -geometry::utils::kPi) {
        headingDiff += 2.0f * geometry::utils::kPi;
      }

      if (std::abs(headingDiff) <= halfViewAngle) {
        visibleTrafficLights.push_back(
            std::make_tuple(dist, headingDiff,
                            trafficLights[objPtr->tlIndex]->getLightState()));
      }
    }

    // we want all the traffic lights sorted by distance to the agent
    std::sort(visibleTrafficLights.begin(), visibleTrafficLights.end(),
              [](auto a, auto b) { return std::get<1>(a) < std::get<1>(b); });

    int nTrafficLights = visibleTrafficLights.size();
    for (size_t k = 0; k < std::min(nTrafficLights, maxNumVisibleTLSigns);
         ++k) {
      auto pointData = visibleTrafficLights[k];
      state[statePosCounter++] = std::get<0>(pointData);
      state[statePosCounter++] = std::get<1>(pointData);
      state[statePosCounter++] = std::get<2>(pointData);
    }
    // increment the state counter appropriately if we have fewer than the
    // maximum number of visible traffic lights
    if (nTrafficLights < maxNumVisibleTLSigns) {
      statePosCounter += (maxNumVisibleTLSigns - nTrafficLights) * 3;
    }
  } else {
    statePosCounter += maxNumVisibleTLSigns * 3;
  }

  return state;
}

std::vector<float> Scenario::getVisibleState(Object* sourceObj, float viewAngle,
                                             float viewDist) {
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

std::vector<float> Scenario::getEgoState(Object* obj) {
  std::vector<float> state(4);

  state[0] = obj->getSpeed();

  float sourceHeading = obj->getHeading();
  while (sourceHeading > geometry::utils::kPi)
    sourceHeading -= 2.0f * geometry::utils::kPi;
  while (sourceHeading < -geometry::utils::kPi)
    sourceHeading += 2.0f * geometry::utils::kPi;

  geometry::Vector2D sourcePos = obj->getPosition();

  geometry::Vector2D goalPos = obj->getGoalPosition();

  geometry::Vector2D otherRelativePos = goalPos - sourcePos;
  float dist = otherRelativePos.Norm();

  float otherRelativeHeading = otherRelativePos.Angle();

  float headingDiff = otherRelativeHeading - sourceHeading;
  if (headingDiff > geometry::utils::kPi)
    headingDiff -= 2.0f * geometry::utils::kPi;
  if (headingDiff < -geometry::utils::kPi)
    headingDiff += 2.0f * geometry::utils::kPi;

  state[1] = dist;
  state[2] = headingDiff;
  state[3] = obj->getLength();

  return state;
}

bool Scenario::checkForCollision(const Object* object1, const Object* object2) {
  // note: right now objects are rectangles but this code works for any pair of
  // convex polygons

  // first check for circles collision
  // float dist = Vector2D::dist(object1->getPosition(),
  // object2->getPosition());
  const float dist =
      geometry::Distance(object1->getPosition(), object2->getPosition());
  const float min_dist = object1->getRadius() + object2->getRadius();
  if (dist > min_dist) {
    return false;
  }
  const geometry::ConvexPolygon polygon1 = object1->BoundingPolygon();
  const geometry::ConvexPolygon polygon2 = object2->BoundingPolygon();
  return polygon1.Intersects(polygon2);
}

bool Scenario::checkForCollision(const Object* object,
                                 const geometry::LineSegment* segment) {
  return geometry::Intersects(*segment, object->BoundingPolygon());
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
  float dHeading = (geometry::utils::kPi / 180.0) *
                   (expertHeadings[objID][timeIdx + 1] -
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
  for (geometry::Vector2D stopSign : stopSigns) {
    float radius = 3;
    sf::CircleShape hexagon(radius, 6);
    hexagon.setFillColor(sf::Color::Red);
    hexagon.setPosition(utils::ToVector2f(stopSign));
    target.draw(hexagon, states);
  }
  for (const auto& object : roadObjects) {
    target.draw(*object, states);
    // draw goal destination
    float radius = 2;
    sf::CircleShape ptShape(radius);
    ptShape.setOrigin(radius, radius);
    ptShape.setFillColor(object->color);
    ptShape.setPosition(utils::ToVector2f(object->goalPosition));
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
std::vector<std::shared_ptr<Object>> Scenario::getRoadObjects() {
  return roadObjects;
}
std::vector<std::shared_ptr<RoadLine>> Scenario::getRoadLines() {
  return roadLines;
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

ImageMatrix Scenario::getCone(Object* object, float viewAngle, float viewDist,
                              float headTilt,
                              bool obscuredView) {  // args in radians
  float circleRadius = viewDist;
  float renderedCircleRadius = 300.0f;

  if (object->coneTexture == nullptr) {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 1;

    object->coneTexture = new sf::RenderTexture();
    object->coneTexture->create(2.0f * renderedCircleRadius,
                                2.0f * renderedCircleRadius, settings);
  }
  sf::RenderTexture* texture = object->coneTexture;

  sf::Transform renderTransform;
  renderTransform.scale(1, -1);  // horizontal flip

  geometry::Vector2D center = object->getPosition();

  texture->clear(sf::Color(50, 50, 50));
  sf::View view(utils::ToVector2f(center, /*flip_y=*/true),
                sf::Vector2f(2.0f * circleRadius, 2.0f * circleRadius));
  view.rotate(-geometry::utils::Degrees(object->getHeading()) + 90.0f);
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

  renderTransform.rotate(-geometry::utils::Degrees(object->getHeading()) +
                         90.0f);

  // TODO(ev) do this for road objects too
  // draw obstructions
  if (obscuredView == true) {
    std::vector<std::shared_ptr<Vehicle>> roadObjects =
        getVehicles();  // todo optimize with objects in range only (quadtree?)

    for (const auto& obj : roadObjects) {
      if (obj.get() != object && obj->occludes) {
        auto lines = obj->getLines();
        for (const auto& [pt1, pt2] : lines) {
          int nIntersections = 0;
          for (const auto& [pt3, pt4] : lines) {
            if (pt1 != pt3 && pt1 != pt4 &&
                geometry::LineSegment(pt1, center)
                    .Intersects(geometry::LineSegment(pt3, pt4))) {
              nIntersections++;
              break;
            }
          }
          for (const auto& [pt3, pt4] : lines) {
            if (pt2 != pt3 && pt2 != pt4 &&
                geometry::LineSegment(pt2, center)
                    .Intersects(geometry::LineSegment(pt3, pt4))) {
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

  renderTransform.rotate(geometry::utils::Degrees(object->getHeading()) -
                         90.0f);
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

ImageMatrix Scenario::getImage(Object* object, bool renderGoals) {
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
    view.rotate(-geometry::utils::Degrees(object->getHeading()) + 90.0f);
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
        ptShape.setFillColor(obj->color);
        ptShape.setPosition(utils::ToVector2f(obj->goalPosition));
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
      ptShape.setFillColor(object->color);
      ptShape.setPosition(utils::ToVector2f(object->goalPosition));
      texture->draw(ptShape, renderTransform);
    }
  }
  for (const auto& obj : roadLines) {
    texture->draw(*obj, renderTransform);
  }
  for (const auto& obj : trafficLights) {
    texture->draw(*obj, renderTransform);
  }
  for (geometry::Vector2D stopSign : stopSigns) {
    float radius = 3;
    sf::CircleShape hexagon(radius, 6);
    hexagon.setFillColor(sf::Color::Red);
    hexagon.setPosition(utils::ToVector2f(stopSign));
    texture->draw(hexagon, renderTransform);
  }
  // render texture and return
  texture->display();

  sf::Image img = texture->getTexture().copyToImage();
  unsigned char* pixelsArr = (unsigned char*)img.getPixelsPtr();

  return ImageMatrix(pixelsArr, squareSide, squareSide, 4);
}

}  // namespace nocturne
