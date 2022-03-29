#pragma once

#include <SFML/Graphics.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "ImageMatrix.hpp"
#include "geometry/bvh.h"
#include "geometry/geometry_utils.h"
#include "geometry/line_segment.h"
#include "object.h"
#include "road.h"
#include "stop_sign.h"
#include "traffic_light.h"
#include "vehicle.h"

namespace pybind11 {

template <typename T, int ExtraFlags>
class array_t;

}  // namespace pybind11

namespace nocturne {

using json = nlohmann::json;

// TODO(ev) hardcoding, this is the maximum number of vehicles that can be
// returned in the state
constexpr int64_t kMaxVisibleKineticObjects = 20;
constexpr int64_t kMaxVisibleRoadPoints = 80;
constexpr int64_t kMaxVisibleTrafficLights = 20;
constexpr int64_t kMaxVisibleStopSigns = 4;

// KineticObject features are:
// [ valid, distance, azimuth, length, witdh, relative_heading,
//   relative_velocity_x, relative_velocity_y, object_type (one_hot of 8) ]
constexpr int64_t kKineticObjectFeatureSize = 16;

// RoadPoint features are:
// [ valid, distance, azimuth, road_type (one_hot of 7) ]
constexpr int64_t kRoadPointFeatureSize = 10;

// TrafficLight features are:
// [ valid, distance, azimuth, current_state (one_hot of 9) ]
constexpr int64_t kTrafficLightFeatureSize = 12;

// StopSign features are:
// [ valid, distance, azimuth ]
constexpr int64_t kStopSignsFeatureSize = 3;

// For py::array_t forward declaration.
// https://github.com/pybind/pybind11/blob/master/include/pybind11/numpy.h#L986
// https://github.com/pybind/pybind11/blob/master/include/pybind11/numpy.h#L143
constexpr int kNumpyArrayForcecast = 0x0010;

class Scenario : public sf::Drawable {
 public:
  Scenario(std::string path, int startTime, bool useNonVehicles);

  void loadScenario(std::string path);

  void step(float dt);
  void waymo_step();  // step forwards and place vehicles at their next position
                      // in the expert dict

  void removeVehicle(Vehicle* object);

  int getMaxEnvTime() { return maxEnvTime; }
  float getSignedAngle(float sourceAngle, float targetAngle) const;

  // query expert data
  std::vector<float> getExpertAction(
      int objID,
      int timeIdx);  // return the expert action of object at time timeIDX
  bool hasExpertAction(
      int objID,
      unsigned int
          timeIdx);  // given the currIndex, figure out if we actually can
                     // compute an expert action given the valid vector
  std::vector<bool> getValidExpertStates(int objID);

  /*********************** State Accessors *******************/
  std::pair<float, geometry::Vector2D> getObjectHeadingAndPos(
      KineticObject* sourceObject);

  sf::FloatRect getRoadNetworkBoundaries() const;

  ImageMatrix getCone(KineticObject* object, float viewDist = 60.0f,
                      float viewAngle = geometry::utils::kHalfPi,
                      float headTilt = 0.0f, bool obscuredView = true);

  ImageMatrix getImage(KineticObject* object = nullptr,
                       bool renderGoals = false);

  bool checkForCollision(const Object& object1, const Object& object2) const;
  bool checkForCollision(const Object& object,
                         const geometry::LineSegment& segment) const;

  std::vector<std::shared_ptr<Vehicle>> getVehicles();
  std::vector<std::shared_ptr<Pedestrian>> getPedestrians();
  std::vector<std::shared_ptr<Cyclist>> getCyclists();
  std::vector<std::shared_ptr<KineticObject>> getRoadObjects();
  std::vector<std::shared_ptr<RoadLine>> getRoadLines();

  std::vector<float> getEgoState(KineticObject* obj);

  pybind11::array_t<float, kNumpyArrayForcecast> Observation(
      const KineticObject& src, float view_dist, float view_angle) const;

  // get a list of vehicles that actually moved
  std::vector<std::shared_ptr<KineticObject>> getObjectsThatMoved() {
    return objectsThatMoved;
  }

 protected:
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

  std::vector<float> ObservationImpl(const KineticObject& src, float view_dist,
                                     float view_angle) const;

  std::tuple<std::vector<const Object*>, std::vector<const Object*>,
             std::vector<const Object*>, std::vector<const Object*>>
  VisibleObjects(const KineticObject& src, float view_dist,
                 float view_angle) const;

  std::vector<const TrafficLight*> VisibleTrafficLights(
      const KineticObject& src, float view_dist, float view_angle) const;

  int currTime;
  int IDCounter = 0;
  int maxEnvTime =
      int(1e5);  // the maximum time an env can run for
                 // set to a big number so that it never overrides the RL env
                 // however, if a traffic light is in the scene then we
                 // set it to 90 so that the episode never runs past
                 // the maximum length of available traffic light data
  bool useNonVehicles;  // used to turn off pedestrians and cyclists

  std::string name;
  std::vector<std::shared_ptr<geometry::LineSegment>> lineSegments;
  std::vector<std::shared_ptr<RoadLine>> roadLines;
  std::vector<std::shared_ptr<Vehicle>> vehicles;
  std::vector<std::shared_ptr<Pedestrian>> pedestrians;
  std::vector<std::shared_ptr<Cyclist>> cyclists;
  std::vector<std::shared_ptr<KineticObject>> roadObjects;
  std::vector<std::shared_ptr<StopSign>> stopSigns;
  std::vector<std::shared_ptr<TrafficLight>> trafficLights;

  sf::RenderTexture* imageTexture = nullptr;
  sf::FloatRect roadNetworkBounds;
  geometry::BVH vehicle_bvh_;       // track vehicles for collisions
  geometry::BVH line_segment_bvh_;  // track line segments for collisions
  geometry::BVH static_bvh_;        // static objects

  // expert data
  std::vector<std::vector<geometry::Vector2D>> expertTrajectories;
  std::vector<std::vector<geometry::Vector2D>> expertSpeeds;
  std::vector<std::vector<float>> expertHeadings;
  std::vector<float> lengths;
  std::vector<std::vector<bool>> expertValid;

  // track the object that moved, useful for figuring out which agents should
  // actually be controlled
  std::vector<std::shared_ptr<KineticObject>> objectsThatMoved;
};

}  // namespace nocturne
