#include "scenario.h"

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

std::vector<const ObjectBase*> VisibleCandidates(const geometry::BVH& bvh,
                                                 const Object& src,
                                                 const ViewField& vf) {
  std::vector<const ObjectBase*> objects =
      bvh.IntersectionCandidates<ObjectBase>(vf);
  auto it = std::find(objects.begin(), objects.end(),
                      dynamic_cast<const ObjectBase*>(&src));
  if (it != objects.end()) {
    std::swap(*it, objects.back());
    objects.pop_back();
  }
  return objects;
}

bool IsVisibleRoadPoint(const Object& src, const ObjectBase& road_point,
                        const std::vector<const ObjectBase*>& objects) {
  // Assume road_point is a point with nearly 0 radius.
  const geometry::LineSegment seg(src.position(), road_point.position());
  for (const ObjectBase* obj : objects) {
    if (obj->can_block_sight() &&
        geometry::Intersects(seg, obj->BoundingPolygon())) {
      return false;
    }
  }
  return true;
}

void VisibleRoadPoints(const Object& src,
                       const std::vector<const ObjectBase*>& objects,
                       std::vector<const ObjectBase*>& road_points) {
  auto pivot =
      std::partition(road_points.begin(), road_points.end(),
                     [&src, &objects](const ObjectBase* road_point) {
                       return IsVisibleRoadPoint(src, *road_point, objects);
                     });
  road_points.resize(std::distance(road_points.begin(), pivot));
}

std::vector<std::pair<const ObjectBase*, float>> NearestK(
    const Object& src, const std::vector<const ObjectBase*>& objects,
    int64_t k) {
  const geometry::Vector2D& src_pos = src.position();
  const int64_t n = objects.size();
  std::vector<std::pair<const ObjectBase*, float>> ret;
  ret.reserve(n);
  for (const ObjectBase* obj : objects) {
    ret.emplace_back(obj, geometry::Distance(src_pos, obj->position()));
  }
  const auto cmp = [](const std::pair<const ObjectBase*, float>& lhs,
                      const std::pair<const ObjectBase*, float>& rhs) {
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

void ExtractObjectFeature(const Object& src, const Object& obj, float dis,
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

void ExtractRoadPointFeature(const Object& src, const RoadPoint& obj, float dis,
                             float* feature) {
  const float azimuth = geometry::utils::AngleSub(
      (obj.position() - src.position()).Angle(), src.heading());
  const int64_t road_type = static_cast<int64_t>(obj.road_type());
  feature[0] = 1.0f;  // Valid
  feature[1] = dis;
  feature[2] = azimuth;
  const geometry::Vector2D neighbor_vec =
      obj.neighbor_position() - obj.position();
  const float neighbor_dis = neighbor_vec.Norm();
  const float neighbor_azimuth =
      geometry::utils::AngleSub(neighbor_vec.Angle(), src.heading());
  feature[3] = neighbor_dis;
  feature[4] = neighbor_azimuth;
  // One-hot vector for road_type, assume feature is initially 0.
  feature[5 + road_type] = 1.0f;
}

void ExtractTrafficLightFeature(const Object& src, const TrafficLight& obj,
                                float dis, float* feature) {
  const float azimuth = geometry::utils::AngleSub(
      (obj.position() - src.position()).Angle(), src.heading());
  const int64_t light_state = static_cast<int64_t>(obj.LightState());
  feature[0] = 1.0f;  // Valid
  feature[1] = dis;
  feature[2] = azimuth;
  // One-hot vector for light_state, assume feature is initially 0.
  feature[3 + light_state] = 1.0f;
}

void ExtractStopSignFeature(const Object& src, const StopSign& obj, float dis,
                            float* feature) {
  const float azimuth = geometry::utils::AngleSub(
      (obj.position() - src.position()).Angle(), src.heading());
  feature[0] = 1.0f;  // Valid
  feature[1] = dis;
  feature[2] = azimuth;
}

}  // namespace

using geometry::utils::kPi;

Scenario::Scenario(const std::string& path, int startTime, bool useNonVehicles)
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
    float width = float(obj["width"]);
    float length = float(obj["length"]);
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

  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();

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
          std::make_shared<StopSign>(position);
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
        min_x = std::min(min_x, std::min(startPoint.x(), endPoint.x()));
        min_y = std::min(min_y, std::min(startPoint.y(), endPoint.y()));
        max_x = std::max(max_x, std::max(startPoint.x(), endPoint.x()));
        max_y = std::max(max_y, std::max(startPoint.y(), endPoint.y()));
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
  road_network_bounds_ =
      sf::FloatRect(min_x, min_y, max_x - min_x, max_y - min_y);

  // Now create the BVH for the line segments
  // Since the line segments never move we only need to define this once
  line_segment_bvh_.InitHierarchy(lineSegments);

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
        std::make_shared<TrafficLight>(geometry::Vector2D(x_pos, y_pos),
                                       validTimes, lightStates, currTime);
    trafficLights.push_back(traffic_light);
  }

  // initialize the road objects bvh
  vehicle_bvh_.InitHierarchy(roadObjects);

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
  // update collision to check for collisions of any vehicles at initialization
  updateCollision();
}

void Scenario::step(float dt) {
  currTime += int(dt / 0.1);  // TODO(ev) hardcoding
  for (auto& object : roadObjects) {
    // reset the collision flags for the objects before stepping
    // we do not want to label a vehicle as persistently having collided
    object->set_collided(false);
    if (!object->expert_control()) {
      object->Step(dt);
    } else {
      geometry::Vector2D expertPosition =
          expertTrajectories[object->id()][currTime];
      object->set_position(expertPosition);
      object->set_heading(expertHeadings[object->id()][currTime]);
    }
  }
  for (auto& object : trafficLights) {
    object->set_current_time(currTime);
  }

  // update the vehicle bvh
  vehicle_bvh_.InitHierarchy(roadObjects);
  updateCollision();
}

void Scenario::updateCollision() {
  // check vehicle-vehicle collisions
  for (auto& obj1 : roadObjects) {
    std::vector<const Object*> candidates =
        vehicle_bvh_.IntersectionCandidates<Object>(*obj1);
    for (const auto* obj2 : candidates) {
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
        const_cast<Object*>(obj2)->set_collided(true);
      }
    }
  }
  // check vehicle-lane segment collisions
  for (auto& obj : roadObjects) {
    std::vector<const geometry::LineSegment*> candidates =
        line_segment_bvh_.IntersectionCandidates<geometry::LineSegment>(*obj);
    for (const auto* seg : candidates) {
      if (checkForCollision(*obj, *seg)) {
        obj->set_collided(true);
      }
    }
  }
}

std::pair<float, geometry::Vector2D> Scenario::getObjectHeadingAndPos(
    Object* sourceObject) {
  float sourceHeading =
      geometry::utils::NormalizeAngle(sourceObject->heading());
  geometry::Vector2D sourcePos = sourceObject->position();
  return std::make_pair(sourceHeading, sourcePos);
}

std::tuple<std::vector<const ObjectBase*>, std::vector<const ObjectBase*>,
           std::vector<const ObjectBase*>, std::vector<const ObjectBase*>>
Scenario::VisibleObjects(const Object& src, float view_dist, float view_angle,
                         float head_tilt) const {
  const float heading = geometry::utils::AngleAdd(src.heading(), head_tilt);
  const geometry::Vector2D& position = src.position();
  const ViewField vf(position, view_dist, heading, view_angle);

  std::vector<const ObjectBase*> objects =
      VisibleCandidates(vehicle_bvh_, src, vf);
  const std::vector<const ObjectBase*> static_candidates =
      VisibleCandidates(static_bvh_, src, vf);
  std::vector<const ObjectBase*> road_points;
  std::vector<const ObjectBase*> traffic_lights;
  std::vector<const ObjectBase*> stop_signs;
  for (const ObjectBase* obj : static_candidates) {
    const StaticObject* obj_ptr = dynamic_cast<const StaticObject*>(obj);
    if (obj_ptr->Type() == StaticObjectType::kRoadPoint) {
      road_points.push_back(dynamic_cast<const ObjectBase*>(obj));
    } else if (obj_ptr->Type() == StaticObjectType::kTrafficLight) {
      traffic_lights.push_back(dynamic_cast<const ObjectBase*>(obj));
    } else if (obj_ptr->Type() == StaticObjectType::kStopSign) {
      stop_signs.push_back(dynamic_cast<const ObjectBase*>(obj));
    }
  }

  vf.FilterVisibleObjects(objects);
  vf.FilterVisiblePoints(road_points);
  VisibleRoadPoints(src, objects, road_points);
  vf.FilterVisibleNonblockingObjects(traffic_lights);
  vf.FilterVisibleNonblockingObjects(stop_signs);

  return std::make_tuple(objects, road_points, traffic_lights, stop_signs);
}

std::vector<const TrafficLight*> Scenario::VisibleTrafficLights(
    const Object& src, float view_dist, float view_angle,
    float head_tilt) const {
  std::vector<const TrafficLight*> ret;

  const float heading = geometry::utils::AngleAdd(src.heading(), head_tilt);
  const geometry::Vector2D& position = src.position();
  const ViewField vf(position, view_dist, heading, view_angle);

  // Assume limited number of TrafficLights, check all of them.
  std::vector<const ObjectBase*> objects;
  objects.reserve(trafficLights.size());
  for (const auto& obj : trafficLights) {
    objects.push_back(dynamic_cast<const ObjectBase*>(obj.get()));
  }
  objects = vf.VisibleNonblockingObjects(objects);

  ret.reserve(objects.size());
  for (const ObjectBase* obj : objects) {
    ret.push_back(dynamic_cast<const TrafficLight*>(obj));
  }

  return ret;
}

NdArray<float> Scenario::EgoState(const Object& src) const {
  NdArray<float> state({kEgoFeatureSize}, 0.0f);

  const float src_heading = src.heading();
  const geometry::Vector2D d = src.destination() - src.position();
  const float dist = d.Norm();
  const float dst_heading = d.Angle();
  const float heading_diff =
      geometry::utils::AngleSub(dst_heading, src_heading);

  float* state_data = state.DataPtr();
  state_data[0] = src.Speed();
  state_data[1] = dist;
  state_data[2] = heading_diff;
  state_data[3] = src.length();
  state_data[4] = src.width();

  return state;
}

std::unordered_map<std::string, NdArray<float>> Scenario::VisibleState(
    const Object& src, float view_dist, float view_angle, float head_tilt,
    bool padding) const {
  const auto [objects, road_points, traffic_lights, stop_signs] =
      VisibleObjects(src, view_dist, view_angle, head_tilt);
  const auto o_targets = NearestK(src, objects, kMaxVisibleObjects);
  const auto r_targets = NearestK(src, road_points, kMaxVisibleRoadPoints);
  const auto t_targets =
      NearestK(src, traffic_lights, kMaxVisibleTrafficLights);
  const auto s_targets = NearestK(src, stop_signs, kMaxVisibleStopSigns);

  const int64_t num_objects =
      padding ? kMaxVisibleObjects : static_cast<int64_t>(o_targets.size());
  const int64_t num_road_points =
      padding ? kMaxVisibleRoadPoints : static_cast<int64_t>(r_targets.size());
  const int64_t num_traffic_lights =
      padding ? kMaxVisibleTrafficLights
              : static_cast<int64_t>(t_targets.size());
  const int64_t num_stop_signs =
      padding ? kMaxVisibleStopSigns : static_cast<int64_t>(s_targets.size());

  NdArray<float> o_feature({num_objects, kObjectFeatureSize}, 0.0f);
  NdArray<float> r_feature({num_road_points, kRoadPointFeatureSize}, 0.0f);
  NdArray<float> t_feature({num_traffic_lights, kTrafficLightFeatureSize},
                           0.0f);
  NdArray<float> s_feature({num_stop_signs, kStopSignsFeatureSize}, 0.0f);

  // Object feature.
  float* o_feature_ptr = o_feature.DataPtr();
  for (const auto [obj, dis] : o_targets) {
    ExtractObjectFeature(src, *(dynamic_cast<const Object*>(obj)), dis,
                         o_feature_ptr);
    o_feature_ptr += kObjectFeatureSize;
  }

  // RoadPoint feature.
  float* r_feature_ptr = r_feature.DataPtr();
  for (const auto [obj, dis] : r_targets) {
    ExtractRoadPointFeature(src, *(dynamic_cast<const RoadPoint*>(obj)), dis,
                            r_feature_ptr);
    r_feature_ptr += kRoadPointFeatureSize;
  }

  // TrafficLight feature.
  float* t_feature_ptr = t_feature.DataPtr();
  for (const auto [obj, dis] : t_targets) {
    ExtractTrafficLightFeature(src, *(dynamic_cast<const TrafficLight*>(obj)),
                               dis, t_feature_ptr);
    t_feature_ptr += kTrafficLightFeatureSize;
  }

  // StopSign feature.
  float* s_feature_ptr = s_feature.DataPtr();
  for (const auto [obj, dis] : s_targets) {
    ExtractStopSignFeature(src, *(dynamic_cast<const StopSign*>(obj)), dis,
                           s_feature_ptr);
    s_feature_ptr += kStopSignsFeatureSize;
  }

  return {{"objects", o_feature},
          {"road_points", r_feature},
          {"traffic_lights", t_feature},
          {"stop_signs", s_feature}};
}

NdArray<float> Scenario::FlattenedVisibleState(const Object& src,
                                               float view_dist,
                                               float view_angle,
                                               float head_tilt) const {
  constexpr int64_t kObjectFeatureStride = 0;
  constexpr int64_t kRoadPointFeatureStride =
      kObjectFeatureStride + kMaxVisibleObjects * kObjectFeatureSize;
  constexpr int64_t kTrafficLightFeatureStride =
      kRoadPointFeatureStride + kMaxVisibleRoadPoints * kRoadPointFeatureSize;
  constexpr int64_t kStopSignFeatureStride =
      kTrafficLightFeatureStride +
      kMaxVisibleTrafficLights * kTrafficLightFeatureSize;
  constexpr int64_t kFeatureSize =
      kStopSignFeatureStride + kMaxVisibleStopSigns * kStopSignsFeatureSize;

  const auto [objects, road_points, traffic_lights, stop_signs] =
      VisibleObjects(src, view_dist, view_angle);

  const auto o_targets = NearestK(src, objects, kMaxVisibleObjects);
  const auto r_targets = NearestK(src, road_points, kMaxVisibleRoadPoints);
  const auto t_targets =
      NearestK(src, traffic_lights, kMaxVisibleTrafficLights);
  const auto s_targets = NearestK(src, stop_signs, kMaxVisibleStopSigns);

  NdArray<float> state({kFeatureSize}, 0.0f);

  // Object feature.
  float* o_feature_ptr = state.DataPtr() + kObjectFeatureStride;
  for (const auto [obj, dis] : o_targets) {
    ExtractObjectFeature(src, *(dynamic_cast<const Object*>(obj)), dis,
                         o_feature_ptr);
    o_feature_ptr += kObjectFeatureSize;
  }

  // RoadPoint feature.
  float* r_feature_ptr = state.DataPtr() + kRoadPointFeatureStride;
  for (const auto [obj, dis] : r_targets) {
    ExtractRoadPointFeature(src, *(dynamic_cast<const RoadPoint*>(obj)), dis,
                            r_feature_ptr);
    r_feature_ptr += kRoadPointFeatureSize;
  }

  // TrafficLight feature.
  float* t_feature_ptr = state.DataPtr() + kTrafficLightFeatureStride;
  for (const auto [obj, dis] : t_targets) {
    ExtractTrafficLightFeature(src, *(dynamic_cast<const TrafficLight*>(obj)),
                               dis, t_feature_ptr);
    t_feature_ptr += kTrafficLightFeatureSize;
  }

  // StopSign feature.
  float* s_feature_ptr = state.DataPtr() + kStopSignFeatureStride;
  for (const auto [obj, dis] : s_targets) {
    ExtractStopSignFeature(src, *(dynamic_cast<const StopSign*>(obj)), dis,
                           s_feature_ptr);
    s_feature_ptr += kStopSignsFeatureSize;
  }

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
  // Update the BVH to account for the fact that some vehicles are now gone
  vehicle_bvh_.InitHierarchy(roadObjects);
}

NdArray<unsigned char> Scenario::getCone(
    Object* object, float viewDist, float viewAngle, float headTilt,
    bool obscuredView) {  // args in radians
  float circleRadius = viewDist;
  constexpr int64_t kRenderedCircleRadius = 300;
  constexpr int64_t kWidth = kRenderedCircleRadius * 2;
  constexpr int64_t kHeight = kRenderedCircleRadius * 2;

  if (object->ConeTexture() == nullptr) {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 1;
    object->InitConeTexture(kHeight, kWidth, settings);
  }
  sf::RenderTexture* texture = object->ConeTexture();

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
  const float r = kRenderedCircleRadius;
  const float diag = std::sqrt(2 * r * r);

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

            float ratio =
                static_cast<float>(kRenderedCircleRadius) / circleRadius;
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

  return NdArray<unsigned char>({kHeight, kWidth, /*channels=*/int64_t(4)},
                                pixelsArr);
}

NdArray<unsigned char> Scenario::getImage(Object* object, bool renderGoals) {
  const int64_t kSquareSide = 600;

  if (image_texture_ == nullptr) {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 4;
    image_texture_ = std::make_unique<sf::RenderTexture>();
    image_texture_->create(kSquareSide, kSquareSide, settings);
  }
  sf::RenderTexture* texture = image_texture_.get();

  sf::Transform renderTransform;
  renderTransform.scale(1, -1);  // horizontal flip

  texture->clear(sf::Color(50, 50, 50));

  // same as in Simulation.cpp
  float padding = 0.0f;
  sf::FloatRect scenarioBounds = road_network_bounds_;
  scenarioBounds.top = -scenarioBounds.top - scenarioBounds.height;
  scenarioBounds.top -= padding;
  scenarioBounds.left -= padding;
  scenarioBounds.width += 2 * padding;
  scenarioBounds.height += 2 * padding;
  sf::Vector2f center =
      sf::Vector2f(scenarioBounds.left + scenarioBounds.width / 2.0f,
                   scenarioBounds.top + scenarioBounds.height / 2.0f);
  sf::Vector2f size =
      sf::Vector2f(kSquareSide, kSquareSide) *
      std::max(scenarioBounds.width / static_cast<float>(kSquareSide),
               scenarioBounds.height / static_cast<float>(kSquareSide));
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

  return NdArray<unsigned char>({kSquareSide, kSquareSide,
                                 /*channels=*/int64_t(4)},
                                pixelsArr);
}

/*********************** Drawing Functions *****************/

sf::View Scenario::View(geometry::Vector2D view_center, float rotation,
                        float view_width, float view_height, float target_width,
                        float target_height, float padding) const {
  // create view (note that the y coordinates and the view rotation are flipped
  // because the scenario is always drawn with a horizontally flip transform)
  const sf::Vector2f center = utils::ToVector2f(view_center, /*flip_y=*/true);
  const sf::Vector2f size(view_width, view_height);
  sf::View view(center, size);
  view.setRotation(-rotation);

  // compute the placement (viewport) of the view within its render target of
  // size (target_width, target_height), so that it is centered with adequate
  // padding and that proportions are keeped (ie. scale-to-fit)
  const float min_ratio = std::min((target_width - 2 * padding) / view_width,
                                   (target_height - 2 * padding) / view_height);
  const float real_view_width_ratio = min_ratio * view_width / target_width;
  const float real_view_height_ratio = min_ratio * view_height / target_height;
  const sf::FloatRect viewport(
      /*left=*/(1.0f - real_view_width_ratio) / 2.0f,
      /*top=*/(1.0f - real_view_height_ratio) / 2.0f,
      /*width=*/real_view_width_ratio,
      /*height=*/real_view_height_ratio);
  view.setViewport(viewport);

  return view;
}

sf::View Scenario::View(float target_width, float target_height,
                        float padding) const {
  // compute center and size of view based on known scenario bounds
  const geometry::Vector2D view_center(
      road_network_bounds_.left + road_network_bounds_.width / 2.0f,
      road_network_bounds_.top + road_network_bounds_.height / 2.0f);
  const float view_width = road_network_bounds_.width;
  const float view_height = road_network_bounds_.height;

  // build the view from overloaded function
  return View(view_center, 0.0f, view_width, view_height, target_width,
              target_height, padding);
}

std::vector<std::unique_ptr<sf::CircleShape>>
Scenario::VehiclesDestinationsDrawables(const Object* source,
                                        float radius) const {
  std::vector<std::unique_ptr<sf::CircleShape>> destination_drawables;
  if (source == nullptr) {
    for (const auto& obj : roadObjects) {
      auto circle_shape = utils::MakeCircleShape(obj->destination(), radius,
                                                 obj->color(), false);
      destination_drawables.push_back(std::move(circle_shape));
    }
  } else {
    auto circle_shape = utils::MakeCircleShape(source->destination(), radius,
                                               source->color(), false);
    destination_drawables.push_back(std::move(circle_shape));
  }
  return destination_drawables;
}

void Scenario::draw(sf::RenderTarget& target, sf::RenderStates states) const {
  sf::Transform horizontal_flip;
  horizontal_flip.scale(1, -1);
  sf::View view =
      View(target.getSize().x, target.getSize().y, /*padding=*/30.0f);
  DrawOnTarget(target, roadLines, view, horizontal_flip);
  DrawOnTarget(target, roadObjects, view, horizontal_flip);
  DrawOnTarget(target, trafficLights, view, horizontal_flip);
  DrawOnTarget(target, stopSigns, view, horizontal_flip);
  DrawOnTarget(target, VehiclesDestinationsDrawables(), view, horizontal_flip);
}

template <typename P>
void Scenario::DrawOnTarget(sf::RenderTarget& target,
                            const std::vector<P>& drawables,
                            const sf::View& view,
                            const sf::Transform& transform) const {
  target.setView(view);
  for (const P& drawable : drawables) {
    target.draw(*drawable, transform);
  }
}

NdArray<unsigned char> Scenario::Image(uint64_t img_width, uint64_t img_height,
                                       bool draw_destinations, float padding,
                                       Object* source, uint64_t view_width,
                                       uint64_t view_height,
                                       bool rotate_with_source) const {
  // construct transform (flip the y-axis)
  sf::Transform horizontal_flip;
  horizontal_flip.scale(1, -1);

  // construct view
  sf::View view;
  if (source == nullptr) {
    // if no source object is provided, get the entire scenario
    view = View(img_width, img_height, padding);
  } else {
    // otherwise get a region around the source object, possibly rotated
    const float rotation =
        rotate_with_source ? geometry::utils::Degrees(source->heading()) - 90.0f
                           : 0.0f;
    view = View(source->position(), rotation, view_width, view_height,
                img_width, img_height, padding);
  }

  // create canvas and draw objects
  Canvas canvas(img_width, img_height);

  DrawOnTarget(canvas, roadLines, view, horizontal_flip);
  DrawOnTarget(canvas, roadObjects, view, horizontal_flip);
  DrawOnTarget(canvas, trafficLights, view, horizontal_flip);
  DrawOnTarget(canvas, stopSigns, view, horizontal_flip);

  if (draw_destinations) {
    DrawOnTarget(canvas, VehiclesDestinationsDrawables(source), view,
                 horizontal_flip);
  }

  return canvas.AsNdArray();
}


NdArray<unsigned char> Scenario::EgoVehicleFeaturesImage(
    const Object& source, float view_dist, float view_angle, float head_tilt,
    uint64_t img_width, uint64_t img_height, float padding, bool draw_source,
    bool draw_destination) const {
  sf::Transform horizontal_flip;
  horizontal_flip.scale(1, -1);

  const float rotation = geometry::utils::Degrees(source.heading()) - 90.0f;
  sf::View view = View(source.position(), rotation, view_dist, view_dist,
                       img_width, img_height, padding);

  Canvas canvas(img_width, img_height);

  // TODO(nl) remove code duplication and linear overhead
  const auto [kinetic_objects, road_points, traffic_lights, stop_signs] =
      VisibleObjects(source, view_dist, view_angle, head_tilt);
  std::vector<const ObjectBase*> drawables;

  for (const auto [objects, kMaxObjects] :
       std::vector<std::pair<std::vector<const ObjectBase*>, int64_t>>{
           {road_points, kMaxVisibleRoadPoints},
           {kinetic_objects, kMaxVisibleObjects},
           {traffic_lights, kMaxVisibleStopSigns},
           {stop_signs, kMaxVisibleTrafficLights},
       }) {
    for (const auto [obj, dist] : NearestK(source, objects, kMaxObjects)) {
      drawables.emplace_back(obj);
    }
  }

  if (draw_source) {
    drawables.emplace_back(&source);
  }
  DrawOnTarget(canvas, drawables, view, horizontal_flip);
  if (draw_destination) {
    DrawOnTarget(canvas, VehiclesDestinationsDrawables(&source), view,
                 horizontal_flip);
  }

  return canvas.AsNdArray();
}

}  // namespace nocturne
