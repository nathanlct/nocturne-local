#include "Scenario.hpp"

#include "geometry/aabb_interface.h"
#include "geometry/line_segment.h"
#include "geometry/box.h"
#include "geometry/vector_2d.h"
#include "utils.hpp"

namespace nocturne {

Scenario::Scenario(std::string path, int startTime, bool useNonVehicles)
    : roadLines(),
      lineSegments(),
      vehicles(),
      cyclists(),
      pedestrians(),
      roadObjects(),
      stopSigns(),
      trafficLights(),
      imageTexture(nullptr),
      expertTrajectories(),
      expertSpeeds(),
      expertHeadings(),
      expertValid(),
      lengths(),
      currTime(startTime),
      useNonVehicles(useNonVehicles) {
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
    for (unsigned int i = 0; i < obj["position"]["x"].size(); i++) {
      geometry::Vector2D currPos(obj["position"]["x"][i],
                                 obj["position"]["y"][i]);
      geometry::Vector2D currVel(obj["velocity"]["x"][i],
                                 obj["velocity"]["y"][i]);
      localExpertTrajectory.push_back(currPos);
      localExpertSpeeds.push_back(currVel);
      localValid.push_back(bool(obj["valid"][i]));
      // waymo data is in degrees!
      float heading = obj["heading"][i];
      localHeadingVec.push_back(heading * float(geometry::utils::kPi / 180.0));
    }
    // TODO(ev) make it a flag whether all vehicles are added or just the
    // vehicles that are valid

    // we only want to store and load vehicles that are valid at this initialization
    // time
    if (bool(obj["valid"][currTime])) {
      expertTrajectories.push_back(localExpertTrajectory);
      expertSpeeds.push_back(localExpertSpeeds);
      expertHeadings.push_back(localHeadingVec);
      lengths.push_back(length);
      expertValid.push_back(localValid);
      if (type == "vehicle") {
        Vehicle* veh = new Vehicle(pos, width, length, heading, occludes,
                                   collides, checkForCollisions, goalPos,
                                   IDCounter,
                                   localExpertSpeeds[currTime].Norm());
        auto ptr = std::shared_ptr<Vehicle>(veh);
        vehicles.push_back(ptr);
        roadObjects.push_back(ptr);
      } else if (type == "pedestrian" && useNonVehicles) {
        Pedestrian* ped = new Pedestrian(pos, width, length, heading, occludes,
                                         collides, checkForCollisions, goalPos,
                                         IDCounter,
                                         localExpertSpeeds[currTime].Norm());
        auto ptr = std::shared_ptr<Pedestrian>(ped);
        pedestrians.push_back(ptr);
        roadObjects.push_back(ptr);
      } else if (type == "cyclist" && useNonVehicles) {
        Cyclist* cyclist = new Cyclist(pos, width, length, heading, occludes,
                                       collides, checkForCollisions, goalPos,
                                       IDCounter,
                                       localExpertSpeeds[currTime].Norm());
        auto ptr = std::shared_ptr<Cyclist>(cyclist);
        cyclists.push_back(ptr);
        roadObjects.push_back(ptr);
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
    bool occludes = false;
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
    }
    // we have to handle stop signs differently from other lane types
    if (road_type == RoadType::stop_sign) {
      stopSigns.push_back(geometry::Vector2D(road["geometry"][0]["x"],
                                             road["geometry"][0]["y"]));
    } else {
      // Iterate over every line segment
      for (int i = 0; i < road["geometry"].size() - 1; i++) {
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

    for (int i = 0; i < tl["state"].size(); i++) {
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
  std::vector<const geometry::AABBInterface*> storeObjects;
  storeObjects.reserve(nRoadObjects);
  for (const auto& obj : roadObjects) {
    storeObjects.push_back(dynamic_cast<const geometry::AABBInterface*>(obj.get()));
  }
  bvh_.InitHierarchy(storeObjects);
  std::cout << "initialized the road objects bvh" << std::endl;

  // Now create the BVH for the inividual road points that will be passed to our agent as state
  // Since the line segments never move we only need to define this once
  const int64_t nLines = roadLines.size();
  const int64_t nPointsPerLine = roadLines[0]->getRoadPoints().size();
  std::vector<const geometry::AABBInterface*> roadPointObjects;
  roadPointObjects.reserve(nLines * nPointsPerLine);
  for (const auto& roadLine : roadLines) {
    for (const auto& roadPoint : roadLine->getRoadPoints()){
      roadPointObjects.push_back(dynamic_cast<const geometry::AABBInterface*>(roadPoint.get()));
    }
  }
  road_point_bvh.InitHierarchy(roadPointObjects);
  std::cout << "initialized the road points bvh" << std::endl;

  // // Now create the BVH for the stop signs
  // // Since the stop signs never move we only need to define this once
  // const int64_t nStopSigns = stopSigns.size();
  // std::vector<const geometry::AABBInterface*> stopSignObjects;
  // stopSignObjects.reserve(nStopSigns);
  // std::cout << "made it to stop sign for loop" << std::endl;
  // for (const auto& obj : stopSigns) {
  //   // create a box to store
  //   geometry::Box* box = new geometry::Box(obj + 2, obj - 2);
  //   std::cout << "initialized stop sign box" << std::endl;
  //   auto ptr = std::shared_ptr<geometry::Box>(box);
  //   stopSignObjects.push_back(dynamic_cast<const geometry::AABBInterface*>(box));
  //   std::cout << "the stop sign area is " << ptr->GetAABB().Area() << std::endl;
  // }
  // std::cout << "exited the stop sign for loop" << std::endl;
  // stop_sign_bvh.InitHierarchy(stopSignObjects);
  // std::cout << "initialized the stop sign bvh" << std::endl;

  // // Now create the BVH for the traffic lights
  // // Since the line segments never move we only need to define this once but
  // // we also need to push back an index into the traffic light states so that 
  // // we can 
  // const int64_t nTrafficLights = trafficLights.size();
  // std::vector<const geometry::AABBInterface*> tlObjects;
  // tlObjects.reserve(nTrafficLights);
  // int i = 0;
  // for (const auto& obj : trafficLights) {
  //   TrafficLightBox* tlBox = new TrafficLightBox(obj->getPosition(), i);
  //   auto ptr = std::shared_ptr<TrafficLightBox>(tlBox);
  //   tlObjects.push_back(dynamic_cast<const geometry::AABBInterface*>(ptr.get()));
  //   i++;
  // }
  // tl_bvh_.InitHierarchy(tlObjects);
  // std::cout << "initialized the traffic light bvh" << std::endl;

}

// void Scenario::createVehicle(float posX, float posY, float width, float length,
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
  std::vector<const geometry::AABBInterface*> objects;
  objects.reserve(n);
  for (const auto& obj : roadObjects) {
    objects.push_back(dynamic_cast<const geometry::AABBInterface*>(obj.get()));
  }
  bvh_.InitHierarchy(objects);
  // check vehicle-vehicle collisions
  for (auto& obj1 : roadObjects) {
    std::vector<const geometry::AABBInterface*> candidates =
        bvh_.IntersectionCandidates(*obj1);
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
  if (currTime < 91){
    currTime += 1;  // TODO(ev) hardcoding
    for (auto& object : roadObjects) {
      geometry::Vector2D expertPosition = expertTrajectories[int(object->getID())][currTime];
      object->setPosition(expertPosition.x(), expertPosition.y());
      object->setHeading(expertHeadings[int(object->getID())][currTime]);
    }
    for (auto& object : trafficLights) {
      object->updateTime(currTime);
    }

    // initalize the vehicle bvh
    const int64_t n = roadObjects.size();
    std::vector<const geometry::AABBInterface*> objects;
    objects.reserve(n);
    for (const auto& obj : roadObjects) {
      objects.push_back(dynamic_cast<const geometry::AABBInterface*>(obj.get()));
    }
    bvh_.InitHierarchy(objects);
    // check vehicle-vehicle collisions
    for (auto& obj1 : roadObjects) {
      std::vector<const geometry::AABBInterface*> candidates =
          bvh_.IntersectionCandidates(*obj1);
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

std::vector<float> Scenario::getVisibleObjectsState(Object* sourceObj, float viewAngle) {
    // TODO(ev) hardcoded
    float viewDist = 120.0f;
    float halfViewAngle = viewAngle / 2.0;

    // Vehicle state is: dist, heading-diff, speed, length
    // TL state is: dist, heading-diff, current light color as an int
    // Road point state is: dist, heading-diff, type of road point as an int i.e. roadEdge, laneEdge, etc.
    // Stop sign state is: dist, heading-diff
    std::vector<float> state(maxNumVisibleVehicles * 4 + maxNumTLSigns * 3 + maxNumVisibleRoadPoints * 3 + 2 * maxNumVisibleStopSigns);
    // TODO(ev) hardcoding
    std::fill(state.begin(), state.end(), -100);
    std::vector<std::tuple<const Object*, float, float>> visibleVehicles;
    int statePosCounter = -1; // set to -1 so the ++ logic for state incrementing works

    // re-center between -pi and pi
    // TODO(ev) this won't work if we are > 2pi or < -2pi
    float sourceHeading = sourceObj->getHeading();
    while (sourceHeading > geometry::utils::kPi)
        sourceHeading -= 2.0f * geometry::utils::kPi;
    while (sourceHeading < -geometry::utils::kPi)
        sourceHeading += 2.0f * geometry::utils::kPi;

    geometry::Vector2D sourcePos = sourceObj->getPosition();

    // TODO(ev) replace this check with an efficient check, this is too large a bound
    // Find a set of plausible candidates that could be in view
    // Do this by bounding the cone in the square that encloses the entire circle that the cone is a part of
    const geometry::Box* outerBox;
    float leftAngle = sourceHeading + geometry::utils::kPi / 4.0f; // the angle pointing to the top left corner of the rectangle enclosing the cone
    float scalingFactorLeft = viewDist / std::cos(halfViewAngle); // how long the vector should be
    geometry::Vector2D topLeft = geometry::Vector2D(std::cos(leftAngle), std::sin(leftAngle));
    topLeft *= scalingFactorLeft;
    topLeft += sourcePos;
    float scalingFactorRight = viewDist / std::sin((geometry::utils::kPi / 2.0) - halfViewAngle);
    float rightAngle = sourceHeading - 3 * geometry::utils::kPi / 4.0f; // the angle pointing to the bottom right hand corner of the rectangle enclosing the cone
    geometry::Vector2D bottomRight = geometry::Vector2D(std::cos(rightAngle), std::sin(rightAngle));
    bottomRight *= scalingFactorRight;
    bottomRight += sourcePos;
    outerBox = new geometry::Box(topLeft, bottomRight);

    //**************** Vehicles **********************
    std::vector<const geometry::AABBInterface*> roadObjCandidates = bvh_.IntersectionCandidates(*outerBox);
    for (const auto* ptr : roadObjCandidates) {
        const Object* objPtr = dynamic_cast<const Object*>(ptr);
        if (objPtr->getID() == sourceObj->getID()){
            continue;
        }
        geometry::Vector2D otherRelativePos = objPtr->getPosition() - sourcePos;
        float dist = otherRelativePos.Norm();

        if (dist > viewDist){
            continue;
        }

        float otherRelativeHeading = otherRelativePos.Angle();

        float headingDiff = otherRelativeHeading - sourceHeading;
        if (headingDiff > geometry::utils::kPi) {headingDiff -= 2.0f * geometry::utils::kPi;}
        if (headingDiff < -geometry::utils::kPi) {headingDiff += 2.0f * geometry::utils::kPi;}

        if (std::abs(headingDiff) <= halfViewAngle) {
            visibleVehicles.push_back(std::make_tuple(objPtr, dist, headingDiff));
        }
    }

    // we want all the vehicles sorted by distance to the agent
    std::sort(
        visibleVehicles.begin(), 
        visibleVehicles.end(),
        [](auto a, auto b) { return std::get<1>(a) < std::get<1>(b); }
    );

    int nVeh = visibleVehicles.size();
    std::cout << "number of visible vehicles is " + std::to_string(nVeh) << std::endl;
    for (size_t k = 0; k < std::min(nVeh, maxNumVisibleVehicles); ++k) {
        auto vehData = visibleVehicles[k];
        state[statePosCounter++] = std::get<1>(vehData);
        state[statePosCounter++] = std::get<2>(vehData);
        state[statePosCounter++] = std::get<0>(vehData)->getSpeed();
        state[statePosCounter++] = std::get<0>(vehData)->getLength();
    }

    //**************** ROAD EDGES **********************
    // Okay now lets run the same process with road edges
    //****************************************
    std::vector<std::tuple<const RoadPoint*, float, float>> visibleRoadPoints;
    std::vector<const geometry::AABBInterface*> roadPointCandidates = road_point_bvh.IntersectionCandidates(*outerBox);
    for (const auto* ptr : roadPointCandidates) {
        const RoadPoint* objPtr = dynamic_cast<const RoadPoint*>(ptr);
        geometry::Vector2D otherRelativePos = objPtr->position - sourcePos;
        float dist = otherRelativePos.Norm();

        if (dist > viewDist){
            continue;
        }

        float otherRelativeHeading = otherRelativePos.Angle();

        float headingDiff = otherRelativeHeading - sourceHeading;
        if (headingDiff > geometry::utils::kPi) {headingDiff -= 2.0f * geometry::utils::kPi;}
        if (headingDiff < -geometry::utils::kPi) {headingDiff += 2.0f * geometry::utils::kPi;}

        if (std::abs(headingDiff) <= halfViewAngle) {
            visibleRoadPoints.push_back(std::make_tuple(objPtr, dist, headingDiff));
        }
    }
    std::cout << "created a list of visible road points" << std::endl;
    // we want all the road points sorted by distance to the agent
    std::sort(
        visibleRoadPoints.begin(), 
        visibleRoadPoints.end(),
        [](auto a, auto b) { return std::get<1>(a) < std::get<1>(b); }
    );
    int nRoadPoints = visibleRoadPoints.size();
    std::cout << "got " + std::to_string(nRoadPoints) + " road points" << std::endl;
    std::cout << "size of state is " + std::to_string(state.size()) << std::endl;
    for (size_t k = 0; k < std::min(nRoadPoints, maxNumVisibleRoadPoints); ++k) {
        std::cout << statePosCounter << std::endl;
        auto pointData = visibleRoadPoints[k];
        state[statePosCounter++] = std::get<1>(pointData);
        std::cout << "got the first state" << std::endl;
        state[statePosCounter++] = std::get<2>(pointData);
        std::cout << "got the second state" << std::endl;
        std::cout << "dist is " + std::to_string( std::get<1>(pointData)) << std::endl;
        state[statePosCounter++] = std::get<0>(pointData)->type;
        std::cout << "got the third state" << std::endl;
    }

    // //**************** STOP SIGNS **********************
    // // Okay now lets run the same process with stop signs
    // //**************** ************************
    // std::vector<std::tuple<float, float>> visibleStopSigns;
    // std::vector<const geometry::AABBInterface*> stopSignCandidates = stop_sign_bvh.CollisionCandidates(dynamic_cast<const geometry::AABBInterface*>(outerBox));
    // for (const auto* ptr : stopSignCandidates) {
    //     const geometry::Box* objPtr = dynamic_cast<const geometry::Box*>(ptr);
    //     geometry::Vector2D otherRelativePos = ((objPtr->Endpoint0() + objPtr->Endpoint1())/2.0) - sourcePos;
    //     float dist = otherRelativePos.Norm();

    //     if (dist > viewDist){
    //         continue;
    //     }

    //     float otherRelativeHeading = otherRelativePos.Angle();

    //     float headingDiff = otherRelativeHeading - sourceHeading;
    //     if (headingDiff > geometry::utils::kPi) {headingDiff -= 2.0f * geometry::utils::kPi;}
    //     if (headingDiff < -geometry::utils::kPi) {headingDiff += 2.0f * geometry::utils::kPi;}

    //     if (std::abs(headingDiff) <= halfViewAngle) {
    //         visibleStopSigns.push_back(std::make_tuple(dist, headingDiff));
    //     }
    // }

    // // we want all the stop signs sorted by distance to the agent
    // std::sort(
    //     visibleStopSigns.begin(), 
    //     visibleStopSigns.end(),
    //     [](auto a, auto b) { return std::get<1>(a) < std::get<1>(b); }
    // );

    // int nStopSigns = visibleStopSigns.size();
    // for (size_t k = 0; k < std::min(nStopSigns, maxNumVisibleStopSigns); ++k) {
    //     auto pointData = visibleStopSigns[k];
    //     state[statePosCounter++] = std::get<0>(pointData);
    //     state[statePosCounter++] = std::get<1>(pointData);
    // }

    // //**************** Traffic Lights **********************
    // // Okay now lets run the same process with traffic lights
    // //**************** *********************************
    // std::vector<std::tuple<float, float, int>> visibleTrafficLights;
    // std::vector<const geometry::AABBInterface*> tlCandidates = tl_bvh_.CollisionCandidates(dynamic_cast<const geometry::AABBInterface*>(outerBox));
    // for (const auto* ptr : tlCandidates) {
    //     const TrafficLightBox* objPtr = dynamic_cast<const TrafficLightBox*>(ptr);
    //     geometry::Vector2D otherRelativePos = objPtr->position - sourcePos;
    //     float dist = otherRelativePos.Norm();

    //     if (dist > viewDist){
    //         continue;
    //     }

    //     float otherRelativeHeading = otherRelativePos.Angle();

    //     float headingDiff = otherRelativeHeading - sourceHeading;
    //     if (headingDiff > geometry::utils::kPi) {headingDiff -= 2.0f * geometry::utils::kPi;}
    //     if (headingDiff < -geometry::utils::kPi) {headingDiff += 2.0f * geometry::utils::kPi;}

    //     if (std::abs(headingDiff) <= halfViewAngle) {
    //         visibleTrafficLights.push_back(std::make_tuple(dist, headingDiff, trafficLights[objPtr->tlIndex]->getLightState()));
    //     }
    // }

    // // we want all the traffic lights sorted by distance to the agent
    // std::sort(
    //     visibleTrafficLights.begin(), 
    //     visibleTrafficLights.end(),
    //     [](auto a, auto b) { return std::get<1>(a) < std::get<1>(b); }
    // );

    // int nTrafficLights = visibleTrafficLights.size();
    // for (size_t k = 0; k < std::min(nTrafficLights, maxNumTLSigns); ++k) {
    //     auto pointData = visibleTrafficLights[k];
    //     state[statePosCounter++] = std::get<0>(pointData);
    //     state[statePosCounter++] = std::get<1>(pointData);
    //     state[statePosCounter++] = std::get<2>(pointData);
    // }

    // return state;
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
    if (headingDiff > geometry::utils::kPi) headingDiff -= 2.0f * geometry::utils::kPi;
    if (headingDiff < -geometry::utils::kPi) headingDiff += 2.0f * geometry::utils::kPi;

    state[1] = dist;
    state[2] = headingDiff;
    state[3] = obj->getLength();

    return state;
}

bool Scenario::checkForCollision(const Object* object1, const Object* object2) {
  // note: right now objects are rectangles but this code works for any pair of
  // convex polygons

  // first check for circles collision
  float dist =
      geometry::Distance(object1->getPosition(), object2->getPosition());
  float minDist = object1->getRadius() + object2->getRadius();
  if (dist > minDist) {
    return false;
  }

  // then for exact collision

  // if the polygons don't intersect, then there exists a line, parallel
  // to a side of one of the two polygons, that entirely separate the two
  // polygons

  // go over both polygons
  for (const Object* polygon : {object1, object2}) {
    // go over all of their sides
    for (const auto& [p, q] : polygon->getLines()) {
      // vector perpendicular to current polygon line
      geometry::Vector2D normal = (q - p).Rotate(geometry::utils::kPi / 2.0f);
      normal.Normalize();

      // project all corners of polygon 1 onto that line
      // min and max represent the boundaries of the polygon's projection on the
      // line
      double min1 = std::numeric_limits<double>::max();
      double max1 = std::numeric_limits<double>::min();
      for (const geometry::Vector2D& pt : object1->getCorners()) {
        // const double projected = normal.dot(pt);
        const double projected = geometry::DotProduct(normal, pt);
        if (projected < min1) min1 = projected;
        if (projected > max1) max1 = projected;
      }

      // same for polygon 2
      double min2 = std::numeric_limits<double>::max();
      double max2 = std::numeric_limits<double>::min();
      for (const geometry::Vector2D& pt : object2->getCorners()) {
        // const double projected = normal.dot(pt);
        const double projected = geometry::DotProduct(normal, pt);
        if (projected < min2) min2 = projected;
        if (projected > max2) max2 = projected;
      }

      if (max1 < min2 || max2 < min1) {
        // we have a line separating both polygons
        return false;
      }
    }
  }

  // we didn't find any line separating both polygons
  return true;
}

bool Scenario::checkForCollision(const Object* object,
                                 const geometry::LineSegment* segment) {
  // note: right now objects are rectangles but this code works for any pair of
  // convex polygons

  // check if the line intersects with any of the edges
  for (std::pair<geometry::Vector2D, geometry::Vector2D> line :
       object->getLines()) {
    if (segment->Intersects(geometry::LineSegment(line.first, line.second))) {
      return true;
    }
  }

  // Now check if both points are inside the polygon
  bool p1_inside = object->pointInside(segment->Endpoint0());
  bool p2_inside = object->pointInside(segment->Endpoint1());
  if (p1_inside && p2_inside) {
    return true;
  } else {
    return false;
  }
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

bool Scenario::hasExpertAction(int objID, int timeIdx) {
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

  for (auto it = vehicles.begin(); it != vehicles.end();) {
    if ((*it).get() == object) {
      it = vehicles.erase(it);
    } else {
      it++;
    }
  }
}

sf::FloatRect Scenario::getRoadNetworkBoundaries() const {
  return roadNetworkBounds;
}

ImageMatrix Scenario::getCone(Object* object, float viewAngle, float headTilt,
                              bool obscuredView) {  // args in radians
  float circleRadius = object->viewRadius;
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

    float angleShift = quadrant * geometry::utils::kPi / 2.0f;

    geometry::Vector2D corner = geometry::PolarToVector2D(
        diag, geometry::utils::kPi / 4.0f + angleShift);
    outerCircle.push_back(
        sf::Vertex(utils::ToVector2f(corner), sf::Color::Black));

    int nPoints = 20;
    for (int i = 0; i < nPoints; ++i) {
      float angle =
          angleShift + i * (geometry::utils::kPi / 2.0f) / (nPoints - 1);

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
  if (viewAngle < 2.0f * geometry::utils::kPi) {
    std::vector<sf::Vertex> innerCircle;  // todo precompute just once

    innerCircle.push_back(
        sf::Vertex(sf::Vector2f(0.0f, 0.0f), sf::Color::Black));
    float startAngle =
        geometry::utils::kPi / 2.0f + headTilt + viewAngle / 2.0f;
    float endAngle = geometry::utils::kPi / 2.0f + headTilt +
                     2.0f * geometry::utils::kPi - viewAngle / 2.0f;

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
  if (object != nullptr){
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
