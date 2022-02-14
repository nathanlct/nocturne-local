#pragma once

#include <SFML/Graphics.hpp>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ImageMatrix.hpp"
#include "Object.hpp"
#include "Road.hpp"
#include "Vehicle.hpp"
#include "geometry/bvh.h"
#include "geometry/geometry_utils.h"
#include "json.hpp"

namespace nocturne {

using json = nlohmann::json;

class Scenario : public sf::Drawable {
 public:
  Scenario(std::string path);

  void loadScenario(std::string path);

  void step(float dt);

  std::vector<std::shared_ptr<Object>> getRoadObjects();
  std::vector<std::shared_ptr<Vehicle>> getVehicles();

  void removeObject(Object* object);

  sf::FloatRect getRoadNetworkBoundaries() const;

  ImageMatrix getCone(
      Object* object,
      float viewAngle = static_cast<float>(geometry::utils::kPi) / 2.0f,
      float headTilt = 0.0f);
  ImageMatrix getImage(Object* object = nullptr, bool renderGoals = false);

  bool checkForCollision(const Object* object1, const Object* object2);

  bool isVehicleOnRoad(const Object& object) const;
  bool isPointOnRoad(float posX, float posY) const;
  void createVehicle(float posX, float posY, float width, float length,
                     float heading, bool occludes, bool collides,
                     bool checkForCollisions, float goalPosX, float goalPosY);

 private:
  virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

  std::string name;
  std::vector<std::shared_ptr<Object>> roadObjects;
  std::vector<std::shared_ptr<Vehicle>> vehicles;
  std::vector<std::shared_ptr<Road>> roads;

  sf::RenderTexture* imageTexture;

  geometry::BVH bvh_;
};

}  // namespace nocturne
