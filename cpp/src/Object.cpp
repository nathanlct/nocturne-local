#include "Object.hpp"

#include "geometry/geometry_utils.h"
#include "utils.hpp"

namespace nocturne {

Object::Object(const geometry::Vector2D& position, float width, float length,
               float heading, bool occludes, bool collides,
               bool checkForCollisions, const geometry::Vector2D& goalPosition,
               int objID, float speed)
    : position(position),
      width(width),
      length(length),
      heading(heading),
      occludes(occludes),
      collides(collides),
      checkForCollisions(checkForCollisions),
      goalPosition(goalPosition),
      id(objID),
      speed(speed),
      type("Object"),
      hasCollided(false),
      coneTexture(nullptr) {
  // generate random color for vehicle
  std::srand(std::time(nullptr) +
             id);  // use current time as seed for random generator
  std::vector<int> colors;
  for (int i = 0; i < 3; i++) {
    colors.push_back(std::rand() / ((RAND_MAX + 1u) / 255));
  }
  colors[std::rand() / ((RAND_MAX + 1u) / 2)] = 255;
  color = sf::Color(colors[0], colors[1], colors[2]);
}

void Object::step(float dt) {}

void Object::draw(sf::RenderTarget& target, sf::RenderStates states) const {
  sf::RectangleShape rect(sf::Vector2f(length, width));
  rect.setOrigin(length / 2.0f, width / 2.0f);
  // rect.setPosition(position.toVector2f());
  rect.setPosition(utils::ToVector2f(position));
  // rect.setRotation(heading * 180 / pi);
  rect.setRotation(geometry::utils::Degrees(heading));

  sf::Color col;
  if (occludes && collides) {
    col = color;  // sf::Color::Red;
  } else if (occludes && !collides) {
    col = sf::Color::Blue;
  } else if (!occludes && collides) {
    col = sf::Color::White;
  } else if (!occludes && !collides) {
    col = sf::Color::Black;
  }

  rect.setFillColor(col);
  target.draw(rect, states);

  // sf::CircleShape circle(3);
  // circle.setOrigin(3, 3);
  // circle.setFillColor(sf::Color::Red);
  // circle.setPosition(position.toVector2f());
  // target.draw(circle, states);

  // for (Vector2D corner : getCorners()) {
  //     sf::CircleShape circle(2);
  //     circle.setOrigin(2, 2);
  //     circle.setFillColor(sf::Color::Red);
  //     circle.setPosition(corner.toVector2f());
  //     target.draw(circle, states);
  // }
}

// Vector2D Object::getPosition() const {
//     return position;
// }
// Vector2D Object::getGoalPosition() const {
//     return goalPosition;
// }
float Object::getSpeed() const { return speed; }
float Object::getHeading() const { return heading; }
float Object::getWidth() const { return width; }
float Object::getLength() const { return length; }
float Object::getRadius() const {
  return std::sqrt(width * width + length * length) / 2.0f;
}
int Object::getID() const { return id; }
std::string Object::getType() const { return type; }

void Object::setSpeed(float newSpeed) { speed = newSpeed; }
void Object::setHeading(float newHeading) { heading = newHeading; }

std::vector<geometry::Vector2D> Object::getCorners() const {
  // Create points
  std::vector<geometry::Vector2D> corners = {
      geometry::Vector2D(length * 0.5, width * 0.5),
      geometry::Vector2D(length * 0.5, -width * 0.5),
      geometry::Vector2D(-length * 0.5, -width * 0.5),
      geometry::Vector2D(-length * 0.5, width * 0.5)};
  // Rotate and translate points
  for (auto& p : corners) {
    p = p.Rotate(heading) + position;
  }
  return corners;
}

std::vector<std::pair<geometry::Vector2D, geometry::Vector2D>>
Object::getLines() const {
  std::vector<geometry::Vector2D> corners = getCorners();
  std::vector<std::pair<geometry::Vector2D, geometry::Vector2D>> lines;
  const int64_t n = corners.size();
  lines.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    lines.emplace_back(corners[i], corners[(i + 1) % n]);
  }
  return lines;
}

bool Object::pointInside(geometry::Vector2D point) const {
  std::vector<geometry::Vector2D> corners = getCorners();
  // check
  int counter = 0;
  int N = corners.size();
  double xinters;

  geometry::Vector2D p1;
  geometry::Vector2D p2;
  p1 = corners[0];
  for (int i = 1; i <= N; i++) {
    p2 = corners[i % N];
    if (point.y() > std::min(p1.y(), p2.y())) {
      if (point.y() <= std::max(p1.y(), p2.y())) {
        if (point.x() <= std::max(p1.x(), p2.x())) {
          if (p1.y() != p2.y()) {
            xinters =
                (point.y() - p1.y()) * (p2.x() - p1.x()) / (p2.y() - p1.y()) +
                p1.x();
            if (p1.x() == p2.x() || point.x() <= xinters) counter++;
          }
        }
      }
    }
    p1 = p2;
  }

  if (counter % 2 == 0)
    return false;
  else
    return true;
}

void Object::setCollided(bool collided) { hasCollided = collided; }

bool Object::getCollided() const { return hasCollided; }

}  // namespace nocturne
