#include "Scenario.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>

#include "geometry/aabb_interface.h"
#include "geometry/geometry_utils.h"
#include "geometry/intersection.h"
#include "geometry/line_segment.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"
#include "utils/sf_utils.h"
#include "view_field.h"

namespace nocturne {

namespace {

std::vector<const Object*> VisibleCandidates(const geometry::BVH& bvh,
                                             const Object& src,
                                             const ViewField& vf) {
  std::vector<const Object*> objects;
  if (!bvh.Empty()) {
    const std::vector<const geometry::AABBInterface*> candidates =
        bvh.IntersectionCandidates(vf);
    for (const auto* obj : candidates) {
      if (obj != &src) {
        objects.push_back(dynamic_cast<const Object*>(obj));
      }
    }
  }
  return objects;
}

bool IsVisibleRoadPoint(const KineticObject& src, const Object& road_point,
                        const std::vector<const Object*>& kinetic_objects) {
  // Assume road_point is a point with nearly 0 radius.
  const geometry::LineSegment seg(src.position(), road_point.position());
  for (const Object* obj : kinetic_objects) {
    if (obj->can_block_sight() &&
        geometry::Intersects(seg, obj->BoundingPolygon())) {
      return false;
    }
  }
  return true;
}

void VisibleRoadPoints(const KineticObject& src,
                       const std::vector<const Object*>& kinetic_objects,
                       std::vector<const Object*>& road_points) {
  auto pivot = std::partition(
      road_points.begin(), road_points.end(),
      [&src, &kinetic_objects](const Object* road_point) {
        return IsVisibleRoadPoint(src, *road_point, kinetic_objects);
      });
  road_points.resize(std::distance(road_points.begin(), pivot));
}

std::vector<std::pair<const Object*, float>> NearestK(
    const Object& src, const std::vector<const Object*>& objects, int64_t k) {
  const geometry::Vector2D& src_pos = src.position();
  const int64_t n = objects.size();
  std::vector<std::pair<const Object*, float>> ret;
  ret.reserve(n);
  for (const Object* obj : objects) {
    ret.emplace_back(obj, geometry::Distance(src_pos, obj->position()));
  }
  const auto cmp = [](const std::pair<const Object*, float>& lhs,
                      const std::pair<const Object*, float>& rhs) {
    return lhs.second < rhs.second;
  };
  if (n <= k) {
    std::sort(ret.begin(), ret.end(), cmp);
  } else {
    std::partial_sort(ret.begin(), ret.begin() + k, ret.end(), cmp);
    ret.resize(k);
  }
  return ret;
}

void ExtractKineticObjectFeature(const KineticObject& src,
                                 const KineticObject& obj, float dis,
                                 float* feature) {
  const float azimuth = geometry::utils::AngleSub(
      (obj.position() - src.position()).Angle(), src.heading());
  const float relative_heading =
      geometry::utils::AngleSub(obj.heading(), src.heading());
  const geometry::Vector2D relative_velocity = obj.velocity() - src.velocity();
  const int64_t obj_type = static_cast<int64_t>(obj.Type());
  feature[0] = 1.0f;  // Valid
  feature[1] = dis;
  feature[2] = azimuth;
  feature[3] = obj.length();
  feature[4] = obj.width();
  feature[5] = relative_heading;
  feature[6] = relative_velocity.Norm();
  feature[7] =
      geometry::utils::AngleSub(relative_velocity.Angle(), src.heading());
  // One-hot vector for object_type, assume feature is initially 0.
  feature[8 + obj_type] = 1.0f;
}

void ExtractRoadPointFeature(const KineticObject& src, const RoadPoint& obj,
                             float dis, float* feature) {
  const float azimuth = geometry::utils::AngleSub(
      (obj.position() - src.position()).Angle(), src.heading());
  const int64_t road_type = static_cast<int64_t>(obj.road_type());
  feature[0] = 1.0f;  // Valid
  feature[1] = dis;
  feature[2] = azimuth;
  // One-hot vector for road_type, assume feature is initially 0.
  feature[3 + road_type] = 1.0f;
}

void ExtractTrafficLightFeature(const KineticObject& src,
                                const TrafficLight& obj, float dis,
                                float* feature) {
  const float azimuth = geometry::utils::AngleSub(
      (obj.position() - src.position()).Angle(), src.heading());
  const int64_t light_state = static_cast<int64_t>(obj.LightState());
  feature[0] = 1.0f;  // Valid
  feature[1] = dis;
  feature[2] = azimuth;
  // One-hot vector for light_state, assume feature is initially 0.
  feature[3 + light_state] = 1.0f;
}

void ExtractStopSignFeature(const KineticObject& src, const StopSign& obj,
                            float dis, float* feature) {
  const float azimuth = geometry::utils::AngleSub(
      (obj.position() - src.position()).Angle(), src.heading());
  feature[0] = 1.0f;  // Valid
  feature[1] = dis;
  feature[2] = azimuth;
}

}  // namespace

using geometry::utils::kPi;

Scenario::Scenario(std::string path, int startTime, bool useNonVehicles)
    : currTime(startTime), useNonVehicles(useNonVehicles) {
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
      if (currVel.Norm() > 0 && bool(obj["valid"][i])) {
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
      // Now finish constructing the entire roadline object which is what will
      // be used for drawing
      int road_size = road["geometry"].size();
      laneGeometry.emplace_back(road["geometry"][road_size - 1]["x"],
                                road["geometry"][road_size - 1]["y"]);
      // TODO(ev) 8 is a hardcoding
      std::shared_ptr<RoadLine> roadLine =
          std::make_shared<RoadLine>(road_type, std::move(laneGeometry),
                                     /*num_road_points=*/8, checkForCollisions);
      roadLines.push_back(roadLine);
    }
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

  std::vector<const geometry::AABBInterface*> static_objects;
  for (const auto& roadLine : roadLines) {
    for (const auto& roadPoint : roadLine->road_points()) {
      static_objects.push_back(
          dynamic_cast<const geometry::AABBInterface*>(&roadPoint));
    }
  }
  for (const auto& obj : stopSigns) {
    static_objects.push_back(
        dynamic_cast<const geometry::AABBInterface*>(obj.get()));
  }
  for (const auto& obj : trafficLights) {
    static_objects.push_back(
        dynamic_cast<const geometry::AABBInterface*>(obj.get()));
  }
  static_bvh_.InitHierarchy(static_objects);
}

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

std::vector<float> Scenario::ObservationImpl(const KineticObject& src,
                                             float view_dist,
                                             float view_angle) const {
  constexpr int64_t kKineticObjectFeatureStride = 0;
  constexpr int64_t kRoadPointFeatureStride =
      kKineticObjectFeatureStride +
      kMaxVisibleKineticObjects * kKineticObjectFeatureSize;
  constexpr int64_t kTrafficLightFeatureStride =
      kRoadPointFeatureStride + kMaxVisibleRoadPoints * kRoadPointFeatureSize;
  constexpr int64_t kStopSignFeatureStride =
      kTrafficLightFeatureStride +
      kMaxVisibleTrafficLights * kTrafficLightFeatureSize;
  constexpr int64_t kFeatureSize =
      kStopSignFeatureStride + kMaxVisibleStopSigns * kStopSignsFeatureSize;

  const auto [kinetic_objects, road_points, traffic_lights, stop_signs] =
      VisibleObjects(src, view_dist, view_angle);

  const auto k_targets =
      NearestK(src, kinetic_objects, kMaxVisibleKineticObjects);
  const auto r_targets = NearestK(src, road_points, kMaxVisibleRoadPoints);
  const auto t_targets =
      NearestK(src, traffic_lights, kMaxVisibleTrafficLights);
  const auto s_targets = NearestK(src, stop_signs, kMaxVisibleStopSigns);

  std::vector<float> obs(kFeatureSize, 0.0f);

  // KineticObject feature.
  float* k_feature_ptr = obs.data() + kKineticObjectFeatureStride;
  for (const auto [obj, dis] : k_targets) {
    ExtractKineticObjectFeature(src, *(dynamic_cast<const KineticObject*>(obj)),
                                dis, k_feature_ptr);
    k_feature_ptr += kKineticObjectFeatureSize;
  }

  // RoadPoint feature.
  float* r_feature_ptr = obs.data() + kRoadPointFeatureStride;
  for (const auto [obj, dis] : r_targets) {
    ExtractRoadPointFeature(src, *(dynamic_cast<const RoadPoint*>(obj)), dis,
                            r_feature_ptr);
    r_feature_ptr += kRoadPointFeatureSize;
  }

  // TrafficLight feature.
  float* t_feature_ptr = obs.data() + kTrafficLightFeatureStride;
  for (const auto [obj, dis] : t_targets) {
    ExtractTrafficLightFeature(src, *(dynamic_cast<const TrafficLight*>(obj)),
                               dis, t_feature_ptr);
    t_feature_ptr += kTrafficLightFeatureSize;
  }

  // StopSign feature.
  float* s_feature_ptr = obs.data() + kStopSignFeatureStride;
  for (const auto [obj, dis] : s_targets) {
    ExtractStopSignFeature(src, *(dynamic_cast<const StopSign*>(obj)), dis,
                           s_feature_ptr);
    s_feature_ptr += kStopSignsFeatureSize;
  }

  return obs;
}

std::tuple<std::vector<const Object*>, std::vector<const Object*>,
           std::vector<const Object*>, std::vector<const Object*>>
Scenario::VisibleObjects(const KineticObject& src, float view_dist,
                         float view_angle) const {
  const float heading = src.heading();
  const geometry::Vector2D& position = src.position();
  const ViewField vf(position, view_dist, heading, view_angle);

  const std::vector<const Object*> static_candidates =
      VisibleCandidates(static_bvh_, src, vf);
  std::vector<const Object*> kinetic_objects =
      VisibleCandidates(vehicle_bvh_, src, vf);
  std::vector<const Object*> road_points;
  std::vector<const Object*> traffic_lights;
  std::vector<const Object*> stop_signs;
  for (const Object* obj : static_candidates) {
    if (obj->Type() == ObjectType::kRoadPoint) {
      road_points.push_back(obj);
    } else if (obj->Type() == ObjectType::kTrafficLight) {
      traffic_lights.push_back(obj);
    } else if (obj->Type() == ObjectType::kStopSign) {
      stop_signs.push_back(obj);
    }
  }

  vf.FilterVisibleObjects(kinetic_objects);
  vf.FilterVisiblePoints(road_points);
  VisibleRoadPoints(src, kinetic_objects, road_points);
  vf.FilterVisibleNonblockingObjects(traffic_lights);
  vf.FilterVisibleNonblockingObjects(stop_signs);

  return std::make_tuple(kinetic_objects, road_points, traffic_lights,
                         stop_signs);
}

std::vector<const TrafficLight*> Scenario::VisibleTrafficLights(
    const KineticObject& src, float view_dist, float view_angle) const {
  std::vector<const TrafficLight*> ret;

  const float heading = src.heading();
  const geometry::Vector2D& position = src.position();
  const ViewField vf(position, view_dist, heading, view_angle);

  // Assume limited number of TrafficLights, check all of them.
  std::vector<const Object*> objects;
  objects.reserve(trafficLights.size());
  for (const auto& obj : trafficLights) {
    objects.push_back(dynamic_cast<const Object*>(obj.get()));
  }
  objects = vf.VisibleNonblockingObjects(objects);

  ret.reserve(objects.size());
  for (const Object* obj : objects) {
    ret.push_back(dynamic_cast<const TrafficLight*>(obj));
  }

  return ret;
}

std::vector<float> Scenario::egoObservationImpl(
    const KineticObject& src) const {
  std::vector<float> state(kEgoFeatureSize);

  state[0] = src.Speed();

  float sourceHeading = geometry::utils::NormalizeAngle(src.heading());

  geometry::Vector2D sourcePos = src.position();
  geometry::Vector2D goalPos = src.destination();

  geometry::Vector2D otherRelativePos = goalPos - sourcePos;
  float dist = otherRelativePos.Norm();

  float otherRelativeHeading = otherRelativePos.Angle();
  float headingDiff = getSignedAngle(sourceHeading, otherRelativeHeading);

  state[1] = dist;
  state[2] = headingDiff;
  state[3] = src.length();
  state[4] = src.width();

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
  for (auto it = objectsThatMoved.begin(); it != objectsThatMoved.end();) {
    if ((*it).get() == object) {
      it = objectsThatMoved.erase(it);
    } else {
      it++;
    }
  }
}

sf::FloatRect Scenario::getRoadNetworkBoundaries() const {
  return roadNetworkBounds;
}

ImageMatrix Scenario::getCone(KineticObject* object, float viewDist,
                              float viewAngle, float headTilt,
                              bool obscuredView) {  // args in radians
  float circleRadius = viewDist;
  float renderedCircleRadius = 300.0f;

  if (object->cone_texture() == nullptr) {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 1;

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

  const auto traffic_lights =
      VisibleTrafficLights(*object, viewDist, viewAngle);
  texture->setView(view);
  for (const auto& tl : traffic_lights) {
    texture->draw(*tl, renderTransform2);
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
        ptShape.setPosition(utils::ToVector2f(obj->destination()));
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
      ptShape.setPosition(utils::ToVector2f(object->destination()));
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
