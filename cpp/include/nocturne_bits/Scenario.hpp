#pragma once

#include <SFML/Graphics.hpp>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ImageMatrix.hpp"
#include "Object.hpp"
#include "RoadLine.hpp"
#include "Vehicle.hpp"
#include "geometry/bvh.h"
#include "geometry/line_segment.h"
#include "geometry/geometry_utils.h"
#include "json.hpp"

namespace nocturne {

using json = nlohmann::json;

class Scenario : public sf::Drawable {
 public:
  Scenario(std::string path);

  void loadScenario(std::string path);
  int currTime = 0; // TODO(ev) this should be passed in rather than defined here

  void step(float dt);

  std::vector<std::shared_ptr<Vehicle>> getVehicles();
  std::vector<std::shared_ptr<RoadLine>> getRoadLines();

  void removeVehicle(Vehicle* object);

  sf::FloatRect getRoadNetworkBoundaries() const;

  ImageMatrix getCone(
      Vehicle* object,
      float viewAngle = static_cast<float>(geometry::utils::kPi) / 2.0f,
      float headTilt = 0.0f,
      bool obscuredView = true);
  ImageMatrix getImage(Object* object = nullptr, bool renderGoals = false);

  bool checkForCollision(const Object* object1, const Object* object2);
  bool checkForCollision(const Object* object, const geometry::LineSegment* segment) ;

  // bool isVehicleOnRoad(const Object& object) const;
  // bool isPointOnRoad(float posX, float posY) const;
  void createVehicle(float posX, float posY, float width, float length,
                     float heading, bool occludes, bool collides,
                     bool checkForCollisions, float goalPosX, float goalPosY);

  // query expert data
  std::vector<float> getExpertAction(int objID, int timeIdx); // return the expert action of object at time timeIDX
  bool hasExpertAction(int objID, int timeIdx); // given the currIndex, figure out if we actually can compute
                                                // an expert action given the valid vector
  std::vector<bool> getValidExpertStates(int objID);

 private:
  virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

  std::string name;
  std::vector<std::shared_ptr<geometry::LineSegment>> lineSegments;
  std::vector<std::shared_ptr<RoadLine>> roadLines;
  std::vector<std::shared_ptr<Vehicle>> vehicles;
  std::vector<geometry::Vector2D> stopSigns;

  sf::RenderTexture* imageTexture;
  sf::FloatRect roadNetworkBounds;
  geometry::BVH bvh_;
  geometry::BVH line_segment_bvh_;

  // expert data
  std::vector<std::vector<geometry::Vector2D>> expertTrajectories;
  std::vector<std::vector<geometry::Vector2D>> expertSpeeds;
  std::vector<std::vector<float>> expertHeadings;
  std::vector<float> lengths;
  std::vector<std::vector<bool>> expertValid;
};

}  // namespace nocturne
