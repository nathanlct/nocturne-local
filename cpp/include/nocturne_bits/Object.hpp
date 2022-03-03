#pragma once

#include <SFML/Graphics.hpp>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"

namespace nocturne {

class Object : public sf::Drawable, public geometry::AABBInterface {
 public:
  Object(const geometry::Vector2D& position, float width, float length,
         float heading, bool occludes, bool collides, bool checkForCollisions,
         const geometry::Vector2D& goalPosition);

  // bool intersectsWith(Object* other) const; // fast spherical pre-check, then
  // accurate rectangular check std::vector<Point> getCorners() const;

  // void move(); // move according to pos, heading and speed
  virtual void step(float dt);

  const geometry::Vector2D& getPosition() const { return position; }

  const geometry::Vector2D& getGoalPosition() const { return goalPosition; }

  float getSpeed() const;
  float getHeading() const;
  float getWidth() const;
  float getLength() const;
  int getID() const;
  std::string getType() const;
  float getRadius() const;  // radius of the minimal circle of center {position}
                            // that includes the whole polygon
  std::vector<geometry::Vector2D> getCorners() const;
  std::vector<std::pair<geometry::Vector2D, geometry::Vector2D>> getLines()
      const;
  geometry::ConvexPolygon BoundingPolygon() const;

  void setPosition(float x, float y) { position = geometry::Vector2D(x, y); }

  void setGoalPosition(float x, float y) {
    goalPosition = geometry::Vector2D(x, y);
  }

  void setSpeed(float speed);
  void setHeading(float heading);

  void setCollided(bool collided);
  bool getCollided() const;

  // TODO: Improve this later.
  geometry::AABB GetAABB() const override {
    const float radius = getRadius();
    return geometry::AABB(getPosition() - radius, getPosition() + radius);
  }

  sf::RenderTexture* coneTexture;

  static int nextID;

 protected:
  virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

  // bool solid;

  geometry::Vector2D position;
  float width;
  float length;
  float heading;

  float speed;
  int id;
  std::string type;

  bool hasCollided;

 public:  // tmp
  bool occludes;
  bool collides;
  bool checkForCollisions;

  nocturne::geometry::Vector2D goalPosition;

  sf::Color color;
};

}  // namespace nocturne
