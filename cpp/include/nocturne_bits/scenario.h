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

constexpr int64_t kMaxEnvTime = 100000;

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
  Scenario(const std::string& scenario_path, int64_t start_time,
           bool allow_non_vehicles)
      : current_time_(start_time), allow_non_vehicles_(allow_non_vehicles) {
    if (!scenario_path.empty()) {
      LoadScenario(scenario_path);
    } else {
      throw std::invalid_argument("No scenario file inputted.");
      // TODO(nl) right now an empty scenario crashes, expectedly
      std::cout << "No scenario path inputted. Defaulting to an empty scenario."
                << std::endl;
    }
  }

  void LoadScenario(const std::string& scenario_path);

  const std::string& name() const { return name_; }

  int64_t max_env_time() const { return max_env_time_; }

  void step(float dt);

  // void removeVehicle(Vehicle* object);
  bool RemoveObject(const Object& object);

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
  // Computes and returns an `sf::View` of size (`view_height`, `view_width`)
  // (in scenario coordinates), centered around `view_center` (in scenario
  // coordinates) and rotated by `rotation` radians. The view is mapped to a
  // viewport of size (`target_height`, `target_width`) pixels, with a minimum
  // padding of `padding` pixels between the scenario boundaries and the
  // viewport border. A scale-to-fit transform is applied so that the scenario
  // view is scaled to fit the viewport (minus padding) without changing the
  // width:height ratio of the captured view.
  sf::View View(geometry::Vector2D view_center, float rotation,
                float view_height, float view_width, float target_height,
                float target_width, float padding) const;

  // Computes and returns an `sf::View``, mapping the whole scenario into a
  // viewport of size (`target_height`, `target_width`) pixels with a minimum
  // padding of `padding` pixels around the scenario. See the other definition
  // of `sf::View View` for more information.
  sf::View View(float target_height, float target_width, float padding) const;

 private:
  // Draws the objects contained in `drawables` on the render target `target`.
  // The view `view` is applied to the target before drawing the objects, and
  // the transform `transform` is applied when drawing each object. `drawables`
  // should contain pointers to objects inheriting from sf::Drawable.
  template <typename P>
  void DrawOnTarget(sf::RenderTarget& target, const std::vector<P>& drawables,
                    const sf::View& view, const sf::Transform& transform) const;

  // Computes and returns a list of `sf::Drawable` objects representing the
  // goals/destinations of the `source` vehicle, or of all vehicles in the
  // scenario if `source == nullptr`. Each goal is represented as a circle of
  // radius `radius`.
  std::vector<std::unique_ptr<sf::CircleShape>> VehiclesDestinationsDrawables(
      const Object* source = nullptr, float radius = 2.0f) const;

  // Draws the scenario to a render target. This is used by SFML to know how
  // to draw classes inheriting sf::Drawable.
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

 public:
  // Computes and returns an image of the scenario. The returned image has
  // dimension `img_height` * `img_width` * 4 where 4 is the number of channels
  // (RGBA). If `draw_destinations` is true, the vehicles' goals will be drawn.
  // `padding` (in pixels) can be used to add some padding around the image
  // (included in its width/height). If a `source` object is provided, computes
  // an image of a rectangle of size (`view_height`, `view_width`) centered
  // around the object, rather than of the whole scenario. Besides, if
  // `rotate_with_source` is set to true, the source object will be pointing
  // upwards (+pi/2) in the returned image. Note that the size of the view will
  // be scaled to fit the image size without changing the width:height ratio, so
  // that the resulting image is not distorted.
  NdArray<unsigned char> Image(uint64_t img_height = 1000,
                               uint64_t img_width = 1000,
                               bool draw_destinations = true,
                               float padding = 0.0f, Object* source = nullptr,
                               uint64_t view_height = 200,
                               uint64_t view_width = 200,
                               bool rotate_with_source = true) const;

  // Computes and returns an image of the visible state of the `source` object,
  // ie. the features returned by the `VisibleState` method. See the
  // documentation of `VisibleState` for an explanation of the `view_dist`,
  // `view_angle` and `head_tilt` parameters. See the documentation of `Image`
  // for an explanation of the remaining parameters of this function.
  NdArray<unsigned char> EgoVehicleFeaturesImage(
      const Object& source, float view_dist = 120.0f,
      float view_angle = geometry::utils::kPi * 0.8f, float head_tilt = 0.0f,
      uint64_t img_height = 1000, uint64_t img_width = 1000,
      float padding = 0.0f, bool draw_destination = true) const;

  // Computes and returns an image of a cone of vision of the `source` object.
  // The image is centered around the `source` object, with a cone of vision of
  // radius `view_dist` and of angle `view_angle` (in radians). The cone points
  // upwards (+pi/2) with an optional tilt `head_tilt` (in radians). See the
  // documentation of `Image` for an explanation of the remaining parameters of
  // this function.
  NdArray<unsigned char> EgoVehicleConeImage(
      const Object& source, float view_dist = 120.0f,
      float view_angle = geometry::utils::kPi * 0.8f, float head_tilt = 0.0f,
      uint64_t img_height = 1000, uint64_t img_width = 1000,
      float padding = 0.0f, bool draw_destinations = true) const;

  /*********************** State Accessors *******************/

 public:
  std::pair<float, geometry::Vector2D> getObjectHeadingAndPos(
      Object* sourceObject);

  bool checkForCollision(const Object& object1, const Object& object2) const;
  bool checkForCollision(const Object& object,
                         const geometry::LineSegment& segment) const;

  const std::vector<std::shared_ptr<Vehicle>>& vehicles() const {
    return vehicles_;
  }

  const std::vector<std::shared_ptr<Pedestrian>>& pedestrians() const {
    return pedestrians_;
  }

  const std::vector<std::shared_ptr<Cyclist>>& cyclists() const {
    return cyclists_;
  }

  const std::vector<std::shared_ptr<Object>>& objects() const {
    return objects_;
  }

  const std::vector<std::shared_ptr<Object>>& moving_objects() const {
    return moving_objects_;
  }

  const std::vector<std::shared_ptr<RoadLine>>& getRoadLines() const {
    return roadLines;
  }

  NdArray<float> EgoState(const Object& src) const;

  std::unordered_map<std::string, NdArray<float>> VisibleState(
      const Object& src, float view_dist, float view_angle,
      float head_tilt = 0.0f, bool padding = false) const;

  NdArray<float> FlattenedVisibleState(const Object& src, float view_dist,
                                       float view_angle,
                                       float head_tilt = 0.0f) const;

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
  VisibleObjects(const Object& src, float view_dist, float view_angle,
                 float head_tilt = 0.0f) const;

  std::vector<const TrafficLight*> VisibleTrafficLights(
      const Object& src, float view_dist, float view_angle,
      float head_tilt = 0.0f) const;

  std::string name_;

  int64_t current_time_;
  int64_t max_env_time_ = kMaxEnvTime;
  const bool allow_non_vehicles_ = true;  // Whether to use non vehicle objects.

  std::vector<std::shared_ptr<geometry::LineSegment>> lineSegments;
  std::vector<std::shared_ptr<RoadLine>> roadLines;

  int64_t object_counter_ = 0;
  std::vector<std::shared_ptr<Vehicle>> vehicles_;
  std::vector<std::shared_ptr<Pedestrian>> pedestrians_;
  std::vector<std::shared_ptr<Cyclist>> cyclists_;
  std::vector<std::shared_ptr<Object>> objects_;
  // Rrack the object that moved, useful for figuring out which agents should
  // actually be controlled
  std::vector<std::shared_ptr<Object>> moving_objects_;

  std::vector<std::shared_ptr<StopSign>> stopSigns;
  std::vector<std::shared_ptr<TrafficLight>> trafficLights;

  geometry::BVH object_bvh_;        // track objects for collisions
  geometry::BVH line_segment_bvh_;  // track line segments for collisions
  geometry::BVH static_bvh_;        // static objects

  // expert data
  std::vector<std::vector<geometry::Vector2D>> expertTrajectories;
  std::vector<std::vector<geometry::Vector2D>> expertSpeeds;
  std::vector<std::vector<float>> expertHeadings;
  std::vector<float> lengths;
  std::vector<std::vector<bool>> expertValid;

  std::unique_ptr<sf::RenderTexture> image_texture_ = nullptr;
  sf::FloatRect road_network_bounds_;
};

}  // namespace nocturne
