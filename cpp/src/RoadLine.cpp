#include "RoadLine.hpp"

#include <iostream>

#include "utils/sf_utils.h"

using namespace std;

namespace nocturne {

RoadLine::RoadLine(std::vector<geometry::Vector2D> geometry, RoadType road_type,
                   bool checkForCollisions, int numPoints)
    : geometry(geometry),
      road_type(road_type),
      roadPoints(),
      numPoints(numPoints),
      checkForCollisions(checkForCollisions) {
  setRoadPoints();
  buildRoadLineGraphics();
}

void RoadLine::setRoadPoints() {
  int stepSize = geometry.size() / numPoints;
  // handle case were numPoints is greater than the
  // size of the list
  if (numPoints > geometry.size()) {
    int diff = numPoints - geometry.size();
    for (int i = 0; i < geometry.size(); i++) {
      auto ptr = std::shared_ptr<RoadPoint>(
          new RoadPoint(geometry[i], static_cast<int>(road_type)));
      roadPoints.push_back(ptr);
    }
    for (int i = 0; i < diff; i++) {
      auto ptr = std::shared_ptr<RoadPoint>(
          new RoadPoint(geometry::Vector2D(-100, -100), -1));
      roadPoints.push_back(ptr);
    }
  } else {
    // TODO(ev) actually we need to make sure we include
    // the most extremal points
    // consider using polyline decimation algos
    for (int i = 0; i < numPoints - 1; i++) {
      RoadPoint* r_pt =
          new RoadPoint(geometry[i * stepSize], static_cast<int>(road_type));
      auto ptr = std::shared_ptr<RoadPoint>(r_pt);
      roadPoints.push_back(ptr);
    }
    RoadPoint* r_pt =
        new RoadPoint(geometry.back(), static_cast<int>(road_type));
    auto ptr = std::shared_ptr<RoadPoint>(r_pt);
    roadPoints.push_back(ptr);
  }
}

std::vector<std::shared_ptr<RoadPoint>> RoadLine::getRoadPoints() {
  return roadPoints;
}

RoadType RoadLine::getRoadType() { return road_type; }

sf::Color RoadLine::getColor(RoadType road_type) {
  // TODO(ev) handle striped and so on
  switch (road_type) {
    case RoadType::lane:
      return sf::Color::Yellow;
    case RoadType::road_line:
      return sf::Color::Blue;
    case RoadType::road_edge:
      return sf::Color::Green;
    case RoadType::stop_sign:
      return sf::Color::Red;
    case RoadType::crosswalk:
      return sf::Color::Magenta;
    case RoadType::speed_bump:
      return sf::Color::Cyan;
    default:
      return sf::Color::Transparent;
  }
}

void RoadLine::buildRoadLineGraphics() {
  // build road lines
  roadLines.clear();

  for (int segment = 0; segment < geometry.size(); ++segment) {
    roadLines.push_back(sf::Vertex(utils::ToVector2f(geometry[segment]),
                                   getColor(getRoadType())));
  }
  // this is just for debugging
  // for (int segment = 0; segment < roadPoints.size(); ++segment) {
  //     roadLines.push_back(sf::Vertex(roadPoints[segment].toVector2f(),
  //     getColor(getRoadType())));
  // }
}

void RoadLine::draw(sf::RenderTarget& target, sf::RenderStates states) const {
  target.draw(&roadLines[0], roadLines.size(), sf::LineStrip, states);
}

}  // namespace nocturne
