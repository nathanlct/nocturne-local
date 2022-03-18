#pragma once

#include <SFML/Graphics.hpp>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ImageMatrix.hpp"
#include "RoadLine.hpp"
#include "geometry/box.h"
#include "geometry/bvh.h"
#include "geometry/geometry_utils.h"
#include "geometry/line_segment.h"
#include "json.hpp"
#include "object.h"
#include "traffic_light.h"
#include "vehicle.h"

namespace nocturne {

using json = nlohmann::json;

class Scenario : public sf::Drawable {
 public:
  Scenario(std::string path, int startTime, bool useNonVehicles);

  void loadScenario(std::string path);

  void step(float dt);
  void waymo_step();  // step forwards and place vehicles at their next position
                      // in the expert dict

  // TODO(ev) hardcoding, this is the maximum number of vehicles that can be
  // returned in the state
  int maxNumVisibleObjects = 20;
  int maxNumVisibleRoadPoints = 80;
  int maxNumVisibleStopSigns = 4;
  int maxNumVisibleTLSigns = 20;

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
  // get the box that encloses the view cone
  const geometry::Box* getOuterBox(float sourceHeading,
                                   geometry::Vector2D sourcePos,
                                   float halfViewAngle, float viewDist);
  std::pair<float, geometry::Vector2D> getObjectHeadingAndPos(
      KineticObject* sourceObject);
  sf::FloatRect getRoadNetworkBoundaries() const;
  ImageMatrix getCone(
      KineticObject* object,
      float viewAngle = static_cast<float>(geometry::utils::kPi) / 2.0f,
      float viewDist = 60.0f, float headTilt = 0.0f, bool obscuredView = true);
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
  std::vector<float> getVisibleObjects(KineticObject* sourceObj,
                                       float viewAngle, float viewDist = 60.0f);
  std::vector<float> getVisibleRoadPoints(KineticObject* sourceObj,
                                          float viewAngle,
                                          float viewDist = 60.0f);
  std::vector<float> getVisibleStopSigns(KineticObject* sourceObj,
                                         float viewAngle,
                                         float viewDist = 60.0f);
  std::vector<float> getVisibleTrafficLights(KineticObject* sourceObj,
                                             float viewAngle,
                                             float viewDist = 60.0f);
  std::vector<float> getVisibleState(
      KineticObject* sourceObj,
      float viewAngle /* the total angle subtended by the view cone*/,
      float viewDist = 60.0f /* how many meters forwards the object can see */);
  // get a list of vehicles that actually moved
  std::vector<std::shared_ptr<KineticObject>> getObjectsThatMoved() {
    return objectsThatMoved;
  }

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
  std::vector<std::shared_ptr<KineticObject>> roadObjects;
  std::vector<geometry::Vector2D> stopSigns;
  std::vector<std::shared_ptr<TrafficLight>> trafficLights;

  sf::RenderTexture* imageTexture;
  sf::FloatRect roadNetworkBounds;
  geometry::BVH vehicle_bvh_;       // track vehicles for collisions
  geometry::BVH line_segment_bvh_;  // track line segments for collisions
  geometry::BVH
      tl_bvh_;  // track traffic light states to find visible traffic lights
  geometry::BVH
      road_point_bvh;           // track road points to find visible road points
  geometry::BVH stop_sign_bvh;  // track stop signs to find visible stop signs

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
