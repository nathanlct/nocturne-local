#pragma once

#include <SFML/Graphics.hpp>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ImageMatrix.hpp"
#include "Object.hpp"
#include "RoadLine.hpp"
#include "TrafficLight.hpp"
#include "Vehicle.hpp"
#include "geometry/bvh.h"
#include "geometry/geometry_utils.h"
#include "geometry/line_segment.h"
#include "json.hpp"

namespace nocturne {

using json = nlohmann::json;

class Scenario : public sf::Drawable {
 public:
  Scenario(std::string path, int startTime, bool useNonVehicles);

  void loadScenario(std::string path);

  void step(float dt);
  void waymo_step(); // step forwards and place vehicles at their next position in the expert dict

  std::vector<std::shared_ptr<Vehicle>> getVehicles();
  std::vector<std::shared_ptr<Pedestrian>> getPedestrians();
  std::vector<std::shared_ptr<Cyclist>> getCyclists();
  std::vector<std::shared_ptr<Object>> getRoadObjects();
  std::vector<std::shared_ptr<RoadLine>> getRoadLines();

  // TODO(ev) hardcoding, this is the maximum number of vehicles that can be returned in the state
  int maxNumVisibleVehicles = 20;
  int maxNumVisibleRoadPoints = 80;
  int maxNumVisibleStopSigns = 4;
  int maxNumTLSigns = 20;

  void removeVehicle(Vehicle* object);

  int getMaxEnvTime() { return maxEnvTime; }

  sf::FloatRect getRoadNetworkBoundaries() const;

  ImageMatrix getCone(
      Object* object,
      float viewAngle = static_cast<float>(geometry::utils::kPi) / 2.0f,
      float headTilt = 0.0f, bool obscuredView = true);
  ImageMatrix getImage(Object* object = nullptr, bool renderGoals = false);

  bool checkForCollision(const Object* object1, const Object* object2);
  bool checkForCollision(const Object* object,
                         const geometry::LineSegment* segment);

  // bool isVehicleOnRoad(const Object& object) const;
  // bool isPointOnRoad(float posX, float posY) const;
//   void createVehicle(float posX, float posY, float width, float length,
//                      float heading, bool occludes, bool collides,
//                      bool checkForCollisions, float goalPosX, float goalPosY);

  // query expert data
  std::vector<float> getExpertAction(
      int objID,
      int timeIdx);  // return the expert action of object at time timeIDX
  bool hasExpertAction(
      int objID,
      unsigned int timeIdx);  // given the currIndex, figure out if we actually can
                     // compute an expert action given the valid vector
  std::vector<bool> getValidExpertStates(int objID);

  // methods for handling state
  std::vector<float> getEgoState(Object* obj);
  std::vector<float> getVisibleObjectsState(Object* sourceObj, float viewAngle /* the total angle subtended by the view cone*/);

 private:
  virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

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
  std::vector<geometry::Vector2D> stopSigns;
  std::vector<std::shared_ptr<TrafficLight>> trafficLights;

  sf::RenderTexture* imageTexture;
  sf::FloatRect roadNetworkBounds;
  geometry::BVH bvh_; // track vehicles for collisions
  geometry::BVH line_segment_bvh_; // track line segments for collisions
  geometry::BVH tl_bvh_; // track traffic light states to find visible traffic lights
  geometry::BVH road_point_bvh; // track road points to find visible road points
  geometry::BVH stop_sign_bvh; // track stop signs to find visible stop signs

  // expert data
  std::vector<std::vector<geometry::Vector2D>> expertTrajectories;
  std::vector<std::vector<geometry::Vector2D>> expertSpeeds;
  std::vector<std::vector<float>> expertHeadings;
  std::vector<float> lengths;
  std::vector<std::vector<bool>> expertValid;
};

}  // namespace nocturne
