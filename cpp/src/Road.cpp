#include "Road.hpp"

#include "geometry/geometry_utils.h"
#include "utils.hpp"

namespace nocturne {

Road::Road(const std::vector<geometry::Vector2D>& geometry, int lanes,
           float laneWidth, bool hasLines)
    : geometry(geometry),
      nLanes(lanes),
      laneWidth(laneWidth),
      initialAngleDelta(static_cast<float>(geometry::utils::kPi) / 2.0f),
      finalAngleDelta(static_cast<float>(geometry::utils::kPi) / 2.0f),
      angles(),
      anglesDelta(),
      lanesGeometry(),
      hasLines(hasLines),
      lineTypes() {
  if (hasLines) {
    lineTypes.push_back(LineType::continuous);
    for (int i = 0; i < nLanes - 1; ++i)
      lineTypes.push_back(LineType::stripped);
    lineTypes.push_back(LineType::continuous);
  } else {
    for (int i = 0; i < nLanes + 1; ++i) lineTypes.push_back(LineType::none);
  }

  buildLanes();
  buildRoadGraphics();
}

void Road::buildLanes() {
  angles.clear();
  lanesGeometry.clear();

  float initialAngle = (geometry[1] - geometry[0]).Angle();
  angles.push_back(initialAngle - initialAngleDelta);
  for (int i = 0; i < geometry.size() - 2; ++i) {
    geometry::Vector2D ptA = geometry[i];
    geometry::Vector2D ptB = geometry[i + 1];
    geometry::Vector2D ptC = geometry[i + 2];

    float angleAB = (ptB - ptA).Angle();
    float angleBC = (ptC - ptB).Angle();
    float angleDelta =
        (angleAB + static_cast<float>(geometry::utils::kPi) - angleBC) / 2.0f;

    anglesDelta.push_back(angleDelta);
    angles.push_back(angleAB - angleDelta);
  }

  float finalAngle =
      (geometry[geometry.size() - 1] - geometry[geometry.size() - 2]).Angle();
  angles.push_back(finalAngle - finalAngleDelta);

  for (int i = 0; i < geometry.size(); ++i) {
    geometry::Vector2D pt = geometry[i];
    float angle = angles[i];

    std::vector<geometry::Vector2D> points;

    float modifiedLaneWidth =
        (i == 0 || i == geometry.size() - 1)
            ? laneWidth
            : laneWidth / std::abs(std::sin(anglesDelta[i - 1]));

    for (int lane = 0; lane < nLanes + 1; ++lane) {
      float shift =
          lane * modifiedLaneWidth - nLanes * modifiedLaneWidth / 2.0f;

      // float dx = shift * std::cos(angle);
      // float dy = shift * std::sin(angle);
      // points.push_back(geometry::Vector2D(pt.x - dx, pt.y - dy));
      const geometry::Vector2D d = geometry::PolarToVector2D(shift, angle);
      points.push_back(pt - d);
    }

    lanesGeometry.push_back(points);
  }
}

std::vector<geometry::Vector2D> Road::getRoadPolygon() const {
  // todo compute this just once
  size_t nPoints = 2 * geometry.size();
  std::vector<geometry::Vector2D> roadPolygon(nPoints);
  for (int i = 0; i < lanesGeometry.size(); ++i) {
    roadPolygon[i] = lanesGeometry[i][0];
    roadPolygon[nPoints - i - 1] =
        lanesGeometry[i][lanesGeometry[i].size() - 1];
  }
  return roadPolygon;
}

void Road::buildRoadGraphics() {
  // build lane quads
  laneQuads.clear();
  for (int segment = 0; segment < lanesGeometry.size() - 1; ++segment) {
    for (int lane = 0; lane < nLanes; ++lane) {
      sf::ConvexShape quad;
      quad.setPointCount(4);
      quad.setFillColor(sf::Color::Black);

      quad.setPoint(0, utils::ToVector2f(lanesGeometry[segment][lane]));
      quad.setPoint(1, utils::ToVector2f(lanesGeometry[segment + 1][lane]));
      quad.setPoint(2, utils::ToVector2f(lanesGeometry[segment + 1][lane + 1]));
      quad.setPoint(3, utils::ToVector2f(lanesGeometry[segment][lane + 1]));

      laneQuads.push_back(quad);
    }
  }

  // build road lines
  roadLines.clear();

  for (int lane = 0; lane < nLanes + 1; ++lane) {
    LineType lineType = lineTypes[lane];
    if (lineType == LineType::none) continue;

    float lineLength = 15.0f;

    bool penDown = true;
    float nextLineLength = lineLength;

    for (int segment = 0; segment < lanesGeometry.size() - 1; ++segment) {
      geometry::Vector2D current = lanesGeometry[segment][lane];
      geometry::Vector2D end = lanesGeometry[segment + 1][lane];
      geometry::Vector2D nextCurrent;

      while (current != end) {
        float lengthLeft = (end - current).Norm();

        if (lengthLeft < nextLineLength) {
          nextCurrent = end;
          nextLineLength -= lengthLeft;
        } else {
          nextCurrent = current + (end - current) * nextLineLength / lengthLeft;
          nextLineLength = lineLength;
        }

        if (penDown) {
          roadLines.push_back(
              sf::Vertex(utils::ToVector2f(current), sf::Color::White));
          roadLines.push_back(
              sf::Vertex(utils::ToVector2f(nextCurrent), sf::Color::White));
        }

        if (lineType != LineType::continuous && nextLineLength == lineLength) {
          penDown = !penDown;
        }

        current = nextCurrent;
      }
    }
  }
}

void Road::draw(sf::RenderTarget& target, sf::RenderStates states) const {
  for (const sf::ConvexShape& quad : laneQuads) {
    target.draw(quad, states);
  }
  target.draw(&roadLines[0], roadLines.size(), sf::Lines, states);

  // for (const std::vector<Vector2D>& line : lanesGeometry) {
  //     for (const Vector2D& point : line) {
  //         sf::CircleShape ptShape(3);
  //         ptShape.setOrigin(3, 3);
  //         ptShape.setFillColor(sf::Color::Red);
  //         ptShape.setPosition(point.x, point.y);
  //         target.draw(ptShape, states);
  //     }
  // }
}

}  // namespace nocturne
