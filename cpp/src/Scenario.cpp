#include "Scenario.hpp"

#include "geometry/aabb_interface.h"
#include "geometry/line_segment.h"
#include "geometry/vector_2d.h"
#include "utils.hpp"

namespace nocturne {

Scenario::Scenario(std::string path)
    : roadLines(),
      lineSegments(),
      vehicles(),
      imageTexture(nullptr),
      expertTrajectories(),
      expertSpeeds(),
      expertHeadings(),
      lengths(),
      expertValid() {
    if (path.size() > 0) {
        loadScenario(path);
    } else {
        throw std::invalid_argument("No scenario file inputted.");
        // TODO(nl) right now an empty scenario crashes, expectedly
        std::cout << "No scenario path inputted. Defaulting to an empty scenario." << std::endl;
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
    std::cout << "name is " + name << std::endl;

    for (const auto& obj : j["objects"]) {
        std::string type = obj["type"];
        // TODO(ev) currTime should be passed in rather than defined here
        geometry::Vector2D pos(obj["position"]["x"][currTime], obj["position"]["y"][currTime]);
        float width = obj["width"];
        float length = obj["length"];
        float heading = geometry::utils::Radians(static_cast<float>(obj["heading"][currTime]));

        // TODO(ev) this should be set elsewhere
        bool occludes = true;
        bool collides = true;
        bool checkForCollisions = true;

        geometry::Vector2D goalPos;
        if (obj.contains("goalPosition")) {
            goalPos = geometry::Vector2D(obj["goalPosition"]["x"], obj["goalPosition"]["y"]);
        }
        std::vector<geometry::Vector2D> localExpertTrajectory;
        std::vector<geometry::Vector2D> localExpertSpeeds;
        std::vector<bool> localValid;
        std::vector<float> localHeadingVec;
        for (unsigned int i = 0; i < obj["position"]["x"].size(); i++) {
            geometry::Vector2D currPos(obj["position"]["x"][i], obj["position"]["y"][i]);
            geometry::Vector2D currVel(obj["velocity"]["x"][i], obj["velocity"]["y"][i]);
            localExpertTrajectory.push_back(currPos);
            localExpertSpeeds.push_back(currVel);
            localValid.push_back(bool(obj["valid"][i]));
            localHeadingVec.push_back(obj["heading"][i]);
        }
        expertTrajectories.push_back(localExpertTrajectory);
        expertSpeeds.push_back(localExpertSpeeds);
        expertHeadings.push_back(localHeadingVec);
        lengths.push_back(length);
        expertValid.push_back(localValid);
        // TODO(ev) add support for pedestrians and cyclists
        // TODO(ev) make it a flag whether all vehicles are added or just the vehicles that are valid
        if (type == "vehicle" && bool(obj["valid"][currTime])) {
            Vehicle* veh = new Vehicle(pos, width, length, heading, occludes, collides, checkForCollisions, goalPos,
                                        localExpertSpeeds[currTime].Norm());
            auto ptr = std::shared_ptr<Vehicle>(veh);
            vehicles.push_back(ptr);
        } else {
            std::cerr << "Unknown object type: " << type << std::endl;
        }
    }

    float minX, minY, maxX, maxY;
    bool first = true;

    // TODO(ev) refactor into a roadline object
    for (const auto& road : j["roads"]) {
        std::vector<geometry::Vector2D> laneGeometry;
        std::string type = road["type"];
        bool checkForCollisions = false;
        bool occludes = false;
        RoadType road_type;
        // TODO(ev) this is not good
        if (type == "lane") {
            road_type = RoadType::lane;
        } else if (type == "road_line") {
            road_type = RoadType::road_line;
        } else if (type == "road_edge") {
            road_type = RoadType::road_edge;
            checkForCollisions = true;
        } else if (type == "stop_sign") {
            road_type = RoadType::stop_sign;
        } else if (type == "crosswalk") {
            road_type = RoadType::crosswalk;
        } else if (type == "speed_bump") {
            road_type = RoadType::speed_bump;
        }
        // Iterate over every line segment
        for (int i = 0; i < road["geometry"].size() - 1; i++) {
            laneGeometry.emplace_back(road["geometry"][i]["x"], road["geometry"][i]["y"]);
            geometry::Vector2D startPoint(road["geometry"][i]["x"], road["geometry"][i]["y"]);
            geometry::Vector2D endPoint(road["geometry"][i + 1]["x"], road["geometry"][i + 1]["y"]);
            // track the minimum boundaries
            if (first) {
                minX = maxX = startPoint.x();
                minY = maxY = startPoint.y();
                first = false;
            } else {
                minX = std::min(minX, std::min(startPoint.x(), endPoint.x()));
                maxX = std::max(maxX, std::max(startPoint.x(), endPoint.x()));
                minY = std::min(minY, std::min(startPoint.y(), endPoint.y()));
                maxY = std::max(maxY, std::max(startPoint.y(), endPoint.y()));
            }
            // We only want to store the line segments for collision checking if collisions are possible
            if (checkForCollisions == true) {
                geometry::LineSegment* lineSegment = new geometry::LineSegment(startPoint, endPoint);
                lineSegments.push_back(std::shared_ptr<geometry::LineSegment>(lineSegment));
            }
        }
        // Now construct the entire roadline object which is what will be used for drawing
        int road_size = road["geometry"].size();
        laneGeometry.emplace_back(road["geometry"][road_size - 1]["x"], road["geometry"][road_size - 1]["y"]);
        RoadLine* roadLine = new RoadLine(laneGeometry, road_type, checkForCollisions);
        roadLines.push_back(std::shared_ptr<RoadLine>(roadLine));
    }
    roadNetworkBounds = sf::FloatRect(minX, minY, maxX - minX, maxY - minY);

    // Now create the BVH for the line segments
    // Since the line segments never move we only need to define this once
    const int64_t n = lineSegments.size();
    std::vector<const geometry::AABBInterface*> objects;
    objects.reserve(n);
    for (const auto& obj : lineSegments) {
        objects.push_back(dynamic_cast<const geometry::AABBInterface*>(obj.get()));
    }
    line_segment_bvh_.InitHierarchy(objects);
}

void Scenario::createVehicle(float posX, float posY, float width, float length, float heading, bool occludes,
                             bool collides, bool checkForCollisions, float goalPosX, float goalPosY) {
    Vehicle* veh = new Vehicle(geometry::Vector2D(posX, posY), width, length, heading, occludes, collides,
                               checkForCollisions, geometry::Vector2D(goalPosX, goalPosY));
    auto ptr = std::shared_ptr<Vehicle>(veh);
    vehicles.push_back(ptr);
}

void Scenario::step(float dt) {
    currTime += int(dt / 0.1); // TODO(ev) hardcoding
    for (auto& object : vehicles) {
        object->step(dt);
    }

    // initalize the vehicle bvh
    const int64_t n = vehicles.size();
    std::vector<const geometry::AABBInterface*> objects;
    objects.reserve(n);
    for (const auto& obj : vehicles) {

        objects.push_back(dynamic_cast<const geometry::AABBInterface*>(obj.get()));
    }
    bvh_.InitHierarchy(objects);
    // check vehicle-vehicle collisions
    for (auto& obj1 : vehicles) {
        std::vector<const geometry::AABBInterface*> candidates =
            bvh_.CollisionCandidates(dynamic_cast<geometry::AABBInterface*>(obj1.get()));
        for (const auto* ptr : candidates) {
            const Object* obj2 = dynamic_cast<const Object*>(ptr);
            if (obj1->getID() == obj2->getID()) {
                continue;
            }
            if (!obj1->checkForCollisions && !obj2->checkForCollisions) {
                continue;
            }
            if (!obj1->collides || !obj2->collides) {
                continue;
            }
            if (checkForCollision(obj1.get(), obj2)) {
                obj1->setCollided(true);
                const_cast<Object*>(obj2)->setCollided(true);
            }
        }
    }
    // check vehicle-lane segment collisions
    for (auto& obj1 : vehicles) {
        std::vector<const geometry::AABBInterface*> candidates =
            line_segment_bvh_.CollisionCandidates(dynamic_cast<geometry::AABBInterface*>(obj1.get()));
        for (const auto* ptr : candidates) {
            const geometry::LineSegment* obj2 = dynamic_cast<const geometry::LineSegment*>(ptr);
            if (checkForCollision(obj1.get(), obj2)) {
                obj1->setCollided(true);
            }
        }
    }
}

bool Scenario::checkForCollision(const Object* object1, const Object* object2) {
    // note: right now objects are rectangles but this code works for any pair of
    // convex polygons

    // first check for circles collision
    // float dist = Vector2D::dist(object1->getPosition(),
    // object2->getPosition());
    float dist = geometry::Distance(object1->getPosition(), object2->getPosition());
    float minDist = object1->getRadius() + object2->getRadius();
    if (dist > minDist) {
        return false;
    }

    // then for exact collision

    // if the polygons don't intersect, then there exists a line, parallel
    // to a side of one of the two polygons, that entirely separate the two
    // polygons

    // go over both polygons
    for (const Object* polygon : {object1, object2}) {
        // go over all of their sides
        for (const auto& [p, q] : polygon->getLines()) {
            // vector perpendicular to current polygon line
            geometry::Vector2D normal = (q - p).Rotate(geometry::utils::kPi / 2.0f);
            normal.Normalize();

            // project all corners of polygon 1 onto that line
            // min and max represent the boundaries of the polygon's projection on the
            // line
            double min1 = std::numeric_limits<double>::max();
            double max1 = std::numeric_limits<double>::min();
            for (const geometry::Vector2D& pt : object1->getCorners()) {
                // const double projected = normal.dot(pt);
                const double projected = geometry::DotProduct(normal, pt);
                if (projected < min1) min1 = projected;
                if (projected > max1) max1 = projected;
            }

            // same for polygon 2
            double min2 = std::numeric_limits<double>::max();
            double max2 = std::numeric_limits<double>::min();
            for (const geometry::Vector2D& pt : object2->getCorners()) {
                // const double projected = normal.dot(pt);
                const double projected = geometry::DotProduct(normal, pt);
                if (projected < min2) min2 = projected;
                if (projected > max2) max2 = projected;
            }

            if (max1 < min2 || max2 < min1) {
                // we have a line separating both polygons
                return false;
            }
        }
    }

    // we didn't find any line separating both polygons
    return true;
}

bool Scenario::checkForCollision(const Object* object, const geometry::LineSegment* segment) {
    // note: right now objects are rectangles but this code works for any pair of
    // convex polygons

    // check if the line intersects with any of the edges
    for (std::pair<geometry::Vector2D, geometry::Vector2D> line : object->getLines()){
        if (segment->Intersects(geometry::LineSegment(line.first, line.second))){
            return true;
        }
    }

    // Now check if both points are inside the polygon
    bool p1_inside = object->pointInside(segment->Endpoint0());
    bool p2_inside = object->pointInside(segment->Endpoint1());
    if (p1_inside && p2_inside) {
        return true;
    }
    else{
        return false;
    }


 }

// TODO(ev) make smoother, also maybe return something named so that
// it's clear what's accel and what's steeringAngle
std::vector<float> Scenario::getExpertAction(int objID, int timeIdx) {
    // we want to return accel, steering angle
    // so first we get the accel
    geometry::Vector2D accel_vec = (expertSpeeds[objID][timeIdx + 1] - expertSpeeds[objID][timeIdx - 1]) / 0.2;
    float accel = accel_vec.Norm();
    float speed = expertSpeeds[objID][timeIdx].Norm();
    float dHeading =
        (geometry::utils::kPi / 180) * (expertHeadings[objID][timeIdx + 1] - expertHeadings[objID][timeIdx - 1]) / 0.2;
    float steeringAngle;
    if (speed > 0.0) {
        float temp = dHeading / speed * lengths[objID];
        std::cout << std::to_string(temp) << std::endl;
        std::cout << std::to_string(asin(temp)) << std::endl;
        steeringAngle = asin(dHeading / speed * lengths[objID]);
    } else {
        steeringAngle = 0.0;
    }
    std::vector<float> expertAction = {accel, steeringAngle};
    return expertAction;
};

bool Scenario::hasExpertAction(int objID, int timeIdx) {
    // The user requested too large a point or a point that
    // can't be used for a second order expansion
    if (timeIdx > expertValid[objID].size() - 1 || timeIdx < 1) {
        return false;
    } else if (!expertValid[objID][timeIdx - 1] || !expertValid[objID][timeIdx + 1]) {
        return false;
    } else {
        return true;
    }
}

std::vector<bool> Scenario::getValidExpertStates(int objID) { return expertValid[objID]; }

void Scenario::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    // for (const auto& road : roads) {
    //   target.draw(*road, states);
    // }
    for (const auto& object : roadLines) {
        target.draw(*object, states);
    }
    for (const auto& object : vehicles) {
        target.draw(*object, states);
        if (object->getType() == "Vehicle") {
            // draw goal destination
            float radius = 2;
            sf::CircleShape ptShape(radius);
            ptShape.setOrigin(radius, radius);
            ptShape.setFillColor(object->color);
            ptShape.setPosition(utils::ToVector2f(object->goalPosition));
            target.draw(ptShape, states);
        }
    }
}

std::vector<std::shared_ptr<Vehicle>> Scenario::getVehicles() { return vehicles; }

std::vector<std::shared_ptr<RoadLine>> Scenario::getRoadLines() {return roadLines; }

void Scenario::removeVehicle(Vehicle* object) {
    for (auto it = vehicles.begin(); it != vehicles.end();) {
        if ((*it).get() == object) {
            it = vehicles.erase(it);
        } else {
            it++;
        }
    }

    for (auto it = vehicles.begin(); it != vehicles.end();) {
        if ((*it).get() == object) {
            it = vehicles.erase(it);
        } else {
            it++;
        }
    }
}

sf::FloatRect Scenario::getRoadNetworkBoundaries() const { return roadNetworkBounds; }

ImageMatrix Scenario::getCone(Vehicle* object, float viewAngle,
                              float headTilt, bool obscuredView) {  // args in radians
    float circleRadius = object->viewRadius;
    float renderedCircleRadius = 300.0f;

  if (object->coneTexture == nullptr) {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 1;

    object->coneTexture = new sf::RenderTexture();
    object->coneTexture->create(2.0f * renderedCircleRadius,
                                2.0f * renderedCircleRadius, settings);
  }
  sf::RenderTexture* texture = object->coneTexture;

  sf::Transform renderTransform;
  renderTransform.scale(1, -1);  // horizontal flip

  geometry::Vector2D center = object->getPosition();

  texture->clear(sf::Color(50, 50, 50));
  sf::View view(utils::ToVector2f(center, /*flip_y=*/true),
                sf::Vector2f(2.0f * circleRadius, 2.0f * circleRadius));
  view.rotate(-geometry::utils::Degrees(object->getHeading()) + 90.0f);
  texture->setView(view);

  texture->draw(
      *this,
      renderTransform);  // todo optimize with objects in range only (quadtree?)

  texture->setView(
      sf::View(sf::Vector2f(0.0f, 0.0f), sf::Vector2f(texture->getSize())));

  // draw circle
  float r = renderedCircleRadius;
  float diag = std::sqrt(2 * r * r);

  for (int quadrant = 0; quadrant < 4; ++quadrant) {
    std::vector<sf::Vertex> outerCircle;  // todo precompute just once

    float angleShift = quadrant * geometry::utils::kPi / 2.0f;

    geometry::Vector2D corner = geometry::PolarToVector2D(
        diag, geometry::utils::kPi / 4.0f + angleShift);
    outerCircle.push_back(
        sf::Vertex(utils::ToVector2f(corner), sf::Color::Black));

    int nPoints = 20;
    for (int i = 0; i < nPoints; ++i) {
      float angle =
          angleShift + i * (geometry::utils::kPi / 2.0f) / (nPoints - 1);

      geometry::Vector2D pt = geometry::PolarToVector2D(r, angle);
      outerCircle.push_back(
          sf::Vertex(utils::ToVector2f(pt), sf::Color::Black));
    }

    texture->draw(&outerCircle[0], outerCircle.size(),
                  sf::TriangleFan);  //, renderTransform);
  }

  // draw cone
  if (viewAngle < 2.0f * geometry::utils::kPi) {
    std::vector<sf::Vertex> innerCircle;  // todo precompute just once

    innerCircle.push_back(
        sf::Vertex(sf::Vector2f(0.0f, 0.0f), sf::Color::Black));
    float startAngle =
        geometry::utils::kPi / 2.0f + headTilt + viewAngle / 2.0f;
    float endAngle = geometry::utils::kPi / 2.0f + headTilt +
                     2.0f * geometry::utils::kPi - viewAngle / 2.0f;

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

  renderTransform.rotate(-geometry::utils::Degrees(object->getHeading()) +
                         90.0f);

  // TODO(ev) do this for road objects too
  // draw obstructions
  if (obscuredView == true) {
    std::vector<std::shared_ptr<Vehicle>> roadObjects =
        getVehicles();  // todo optimize with objects in range only (quadtree?)

    for (const auto& obj : roadObjects) {
        if (obj.get() != object && obj->occludes) {
        auto lines = obj->getLines();
        for (const auto& [pt1, pt2] : lines) {
            int nIntersections = 0;
            for (const auto& [pt3, pt4] : lines) {
            if (pt1 != pt3 && pt1 != pt4 &&
                geometry::LineSegment(pt1, center)
                    .Intersects(geometry::LineSegment(pt3, pt4))) {
                nIntersections++;
                break;
            }
            }
            for (const auto& [pt3, pt4] : lines) {
            if (pt2 != pt3 && pt2 != pt4 &&
                geometry::LineSegment(pt2, center)
                    .Intersects(geometry::LineSegment(pt3, pt4))) {
                nIntersections++;
                break;
            }
            }

            if (nIntersections >= 1) {
            sf::ConvexShape hiddenArea;

            float angle1 = (pt1 - center).Angle();
            float angle2 = (pt2 - center).Angle();
            while (angle2 > angle1) angle2 -= 2.0f * geometry::utils::kPi;

            int nPoints = 80;  // todo function of angle
            hiddenArea.setPointCount(nPoints + 2);

            float ratio = renderedCircleRadius / circleRadius;
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

  texture->display();

  sf::Image img = texture->getTexture().copyToImage();
  unsigned char* pixelsArr = (unsigned char*)img.getPixelsPtr();

  return ImageMatrix(pixelsArr, renderedCircleRadius * 2, renderedCircleRadius * 2, 4);
}

ImageMatrix Scenario::getImage(Object* object, bool renderGoals) {
    int squareSide = 600;

    if (imageTexture == nullptr) {
        sf::ContextSettings settings;
        settings.antialiasingLevel = 4;
        imageTexture = new sf::RenderTexture();
        imageTexture->create(squareSide, squareSide, settings);
    }
    sf::RenderTexture* texture = imageTexture;

    sf::Transform renderTransform;
    renderTransform.scale(1, -1);  // horizontal flip

    texture->clear(sf::Color(50, 50, 50));

    // same as in Simulation.cpp
    float padding = 0.0f;
    sf::FloatRect scenarioBounds = getRoadNetworkBoundaries();
    scenarioBounds.top = -scenarioBounds.top - scenarioBounds.height;
    scenarioBounds.top -= padding;
    scenarioBounds.left -= padding;
    scenarioBounds.width += 2 * padding;
    scenarioBounds.height += 2 * padding;
    sf::Vector2f center = sf::Vector2f(scenarioBounds.left + scenarioBounds.width / 2.0f,
                                       scenarioBounds.top + scenarioBounds.height / 2.0f);
    sf::Vector2f size = sf::Vector2f(squareSide, squareSide) *
                        std::max(scenarioBounds.width / squareSide, scenarioBounds.height / squareSide);
    sf::View view(center, size);
    float heading = object->getHeading();
    view.rotate(-geometry::utils::Degrees(object->getHeading()) + 90.0f);

    texture->setView(view);

    // for (const auto& road : roads) {
    //   texture->draw(*road, renderTransform);
    // }
    if (object == nullptr) {
        for (const auto& obj : roadLines) {
            texture->draw(*obj, renderTransform);
        }
        for (const auto& obj : vehicles) {
            texture->draw(*obj, renderTransform);
            if (renderGoals && obj->getType() == "Vehicle") {
                // draw goal destination
                float radius = 2;
                sf::CircleShape ptShape(radius);
                ptShape.setOrigin(radius, radius);
                ptShape.setFillColor(obj->color);
                ptShape.setPosition(utils::ToVector2f(obj->goalPosition));
                texture->draw(ptShape, renderTransform);
            }
        }
    } else {
        texture->draw(*object, renderTransform);

        for (const auto& obj : roadLines) {
            texture->draw(*obj, renderTransform);
        }

        if (renderGoals) {
            // draw goal destination
            float radius = 2;
            sf::CircleShape ptShape(radius);
            ptShape.setOrigin(radius, radius);
            ptShape.setFillColor(object->color);
            ptShape.setPosition(utils::ToVector2f(object->goalPosition));
            texture->draw(ptShape, renderTransform);
        }
    }

    // render texture and return
    texture->display();

    sf::Image img = texture->getTexture().copyToImage();
    unsigned char* pixelsArr = (unsigned char*)img.getPixelsPtr();

    return ImageMatrix(pixelsArr, squareSide, squareSide, 4);
}

}  // namespace nocturne
