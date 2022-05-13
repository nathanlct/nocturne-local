#pragma once

#include <SFML/Graphics.hpp>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "canvas.h"
#include "cyclist.h"
#include "geometry/bvh.h"
#include "geometry/geometry_utils.h"
#include "geometry/line_segment.h"
#include "geometry/vector_2d.h"
#include "ndarray.h"
#include "object.h"
#include "object_base.h"
#include "pedestrian.h"
#include "road.h"
#include "static_object.h"
#include "stop_sign.h"
#include "traffic_light.h"
#include "vehicle.h"

namespace nocturne {

using json = nlohmann::json;

// TODO(ev) hardcoding, this is the maximum number of vehicles that can be
// returned in the state
constexpr int64_t kMaxVisibleObjects = 20;
constexpr int64_t kMaxVisibleRoadPoints = 300;
constexpr int64_t kMaxVisibleTrafficLights = 20;
constexpr int64_t kMaxVisibleStopSigns = 4;

// Object features are:
// [ valid, distance, azimuth, length, witdh, relative_heading,
//   relative_velocity_speed, relative_velocity_direction,
//   object_type (one_hot of 5) ]
constexpr int64_t kObjectFeatureSize = 13;

// RoadPoint features are:
// [ valid, distance, azimuth, distance to the next point, relative azimuth to
//   the next point, road_type (one_hot of 7) ]
constexpr int64_t kRoadPointFeatureSize = 12;

// TrafficLight features are:
// [ valid, distance, azimuth, current_state (one_hot of 9) ]
constexpr int64_t kTrafficLightFeatureSize = 12;

// StopSign features are:
// [ valid, distance, azimuth ]
constexpr int64_t kStopSignsFeatureSize = 3;

// Ego features are:
// [ ego speed, distance to goal position, relative angle to goal position,
// length, width ]
constexpr int64_t kEgoFeatureSize = 5;

class Scenario : public sf::Drawable {
 public:
  Scenario(const std::string& path, int startTime, bool useNonVehicles);

  void loadScenario(std::string path);

  void step(float dt);

  void removeVehicle(Vehicle* object);

  int getMaxEnvTime() { return maxEnvTime; }
  // float getSignedAngle(float sourceAngle, float targetAngle) const;

  // query expert data
  geometry::Vector2D getExpertSpeeds(int timeIndex, int vehIndex) {
    return expertSpeeds[vehIndex][timeIndex];
  };
  std::vector<float> getExpertAction(
      int objID,
      int timeIdx);  // return the expert action of object at time timeIDX
  bool hasExpertAction(
      int objID,
      unsigned int
          timeIdx);  // given the currIndex, figure out if we actually can
                     // compute an expert action given the valid vector
  std::vector<bool> getValidExpertStates(int objID);

  /*********************** Drawing Functions *****************/

 public:
  // Computes and returns an `sf::View` of size (`view_width`, `view_height`)
  // (in scenario coordinates), centered around `view_center` (in scenario
  // coordinates) and rotated by `rotation` radians. The view is mapped to a
  // viewport of size (`target_width`, `target_height`) pixels, with a minimum
  // padding of `padding` pixels between the scenario boundaries and the
  // viewport border. A scale-to-fit transform is applied so that the scenario
  // view is scaled to fit the viewport (minus padding) without changing the
  // width:height ratio of the captured view.
  sf::View View(geometry::Vector2D view_center, float rotation,
                float view_width, float view_height, float target_width,
                float target_height, float padding) const;

  // Computes and returns an `sf::View``, mapping the whole scenario into a
  // viewport of size (`target_width`, `target_height`) pixels with a minimum
  // padding of `padding` pixels around the scenario. See the other definition
  // of `sf::View View` for more information.
  sf::View View(float target_width, float target_height, float padding) const;

 private:
  // Computes and returns a list of `sf::Drawable` objects representing the
  // goals/destinations of the `source` vehicle, or of all vehicles in the
  // scenario if `source == nullptr`. Each goal is represented as a circle of
  // radius `radius`.
  std::vector<std::unique_ptr<sf::CircleShape>> VehiclesDestinationsDrawables(
      Object* source = nullptr, float radius = 2.0f) const;

  // Draws the scenario to a render target. This is used by SFML to know how
  // to draw classes inheriting sf::Drawable.
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

  // Draws the objects contained in `drawables` on the render target `target`.
  // The view `view` is applied to the target before drawing the objects, and
  // the transform `transform` is applied when drawing each object. `drawables`
  // should contain pointers to objects inheriting from sf::Drawable.
  template <typename P>
  void DrawOnTarget(sf::RenderTarget& target, const std::vector<P>& drawables,
                    const sf::View& view, const sf::Transform& transform) const;

public:
  NdArray<unsigned char> Image(uint64_t img_width = 500,
                               uint64_t img_height = 500,
                               bool draw_destinations = true,
                               float padding = 0.0f, Object* source = nullptr,
                               uint64_t view_width = 200,
                               uint64_t view_height = 200,
                               bool rotate_with_source = true) const;

  NdArray<unsigned char> getCone(Object* object, float viewDist = 60.0f,
                                 float viewAngle = geometry::utils::kHalfPi,
                                 float headTilt = 0.0f,
                                 bool obscuredView = true);  // TODO REMOVE

  NdArray<unsigned char> getImage(Object* object = nullptr,
                                  bool renderGoals = false);  // TODO REMOVE

  /*********************** State Accessors *******************/

 public:
  std::pair<float, geometry::Vector2D> getObjectHeadingAndPos(
      Object* sourceObject);

  bool checkForCollision(const Object& object1, const Object& object2) const;
  bool checkForCollision(const Object& object,
                         const geometry::LineSegment& segment) const;

  const std::vector<std::shared_ptr<Vehicle>>& getVehicles() const {
    return vehicles;
  }

  const std::vector<std::shared_ptr<Pedestrian>>& getPedestrians() const {
    return pedestrians;
  }

  const std::vector<std::shared_ptr<Cyclist>>& getCyclists() const {
    return cyclists;
  }

  const std::vector<std::shared_ptr<Object>>& getRoadObjects() const {
    return roadObjects;
  }

  const std::vector<std::shared_ptr<RoadLine>>& getRoadLines() const {
    return roadLines;
  }

  NdArray<float> EgoState(const Object& src) const;

  std::unordered_map<std::string, NdArray<float>> VisibleState(
      const Object& src, float view_dist, float view_angle,
      bool padding = false) const;

  NdArray<float> FlattenedVisibleState(const Object& src, float view_dist,
                                       float view_angle) const;

  // get a list of vehicles that actually moved
  std::vector<std::shared_ptr<Object>> getObjectsThatMoved() {
    return objectsThatMoved;
  }
  int64_t getMaxNumVisibleObjects() const { return kMaxVisibleObjects; }
  int64_t getMaxNumVisibleRoadPoints() const { return kMaxVisibleRoadPoints; }
  int64_t getMaxNumVisibleTrafficLights() const {
    return kMaxVisibleTrafficLights;
  }
  int64_t getMaxNumVisibleStopSigns() const { return kMaxVisibleStopSigns; }
  int64_t getObjectFeatureSize() const { return kObjectFeatureSize; }
  int64_t getRoadPointFeatureSize() const { return kRoadPointFeatureSize; }
  int64_t getTrafficLightFeatureSize() const {
    return kTrafficLightFeatureSize;
  }
  int64_t getStopSignsFeatureSize() const { return kStopSignsFeatureSize; }
  int64_t getEgoFeatureSize() const { return kEgoFeatureSize; }

 protected:
  // update the collision status of all objects
  void updateCollision();

  std::tuple<std::vector<const ObjectBase*>, std::vector<const ObjectBase*>,
             std::vector<const ObjectBase*>, std::vector<const ObjectBase*>>
  VisibleObjects(const Object& src, float view_dist, float view_angle) const;

  std::vector<const TrafficLight*> VisibleTrafficLights(const Object& src,
                                                        float view_dist,
                                                        float view_angle) const;

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
  std::vector<std::shared_ptr<Object>> roadObjects;
  std::vector<std::shared_ptr<StopSign>> stopSigns;
  std::vector<std::shared_ptr<TrafficLight>> trafficLights;

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
  std::vector<std::shared_ptr<Object>> objectsThatMoved;

  std::unique_ptr<sf::RenderTexture> image_texture_ = nullptr;
  sf::FloatRect road_network_bounds_;
};

}  // namespace nocturne
