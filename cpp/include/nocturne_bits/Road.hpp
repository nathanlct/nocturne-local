#pragma once

#include <SFML/Graphics.hpp>
#include <vector>

#include "LineType.hpp"
#include "geometry/vector_2d.h"

namespace nocturne {

class Road : public sf::Drawable {
 public:
  Road(const std::vector<geometry::Vector2D>& geometry, int lanes,
       float laneWidth, bool hasLines);

  std::vector<geometry::Vector2D> getRoadPolygon() const;

 private:
  virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

  void buildLanes();
  void buildRoadGraphics();

  std::vector<geometry::Vector2D> geometry;
  float laneWidth;
  int nLanes;
  bool hasLines;

  float initialAngleDelta;
  float finalAngleDelta;
  std::vector<float> angles;
  std::vector<float> anglesDelta;

  std::vector<LineType> lineTypes;

  // if road has n lanes and m segments, then array of shape (m+1,n+1)
  // where index i,j is the 2D point at the intersection of:
  // - the line separating road segment i from road segment i-1 (indexed from
  // start to end)
  // - the line separating lane j from lane j-1 in road segment i (index from
  // right to left) indexes out of bounds correspond to out of road ^ worst
  // explanation ever
  std::vector<std::vector<geometry::Vector2D>> lanesGeometry;

  std::vector<sf::ConvexShape> laneQuads;
  std::vector<sf::Vertex> roadLines;
};

}  // namespace nocturne
