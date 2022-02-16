#pragma once

#include <vector>
#include <cmath>

#include <SFML/Graphics.hpp>
#include "geometry/vector_2d.h"

namespace nocturne {

enum class RoadType {
    lane,
    road_line,
    road_edge,
    stop_sign,
    crosswalk,
    speed_bump
};

class RoadLine : public sf::Drawable {
public:
    RoadLine(std::vector<geometry::Vector2D> geometry, RoadType road_type,
            bool checkForCollisions=false,
            int numPoints=8);
            // num_points = how many points are returned to 
            // represent the road line
    std::vector<geometry::Vector2D> getSplineCoeffs() const;
    std::vector<geometry::Vector2D> getRoadPolyLine() const;
    void setRoadPoints(); // return an evenly spaced set of the geometry
    std::vector<geometry::Vector2D> getRoadPoints(); // return an evenly spaced set of the geometry
    void setState();
    std::vector<float> getState();
    std::vector<float> getLocalState(geometry::Vector2D vehPos);
    sf::Color getColor(RoadType road_type);
    RoadType getRoadType();

private:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

    void computeSpline();
    void buildRoadLineGraphics();

    std::vector<geometry::Vector2D> geometry;
    std::vector<geometry::Vector2D> splineCoefficients;
    std::vector<geometry::Vector2D> roadPoints; 
    std::vector<float> state; // the state that is returned to an RL agent
    std::vector<sf::Vertex> roadLines;
    bool checkForCollisions;
    RoadType road_type;
    int numPoints; 
};

}  // namespace nocturne
