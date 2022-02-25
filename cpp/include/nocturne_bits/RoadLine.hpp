#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include <SFML/Graphics.hpp>
#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
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

class RoadPoint : public geometry::AABBInterface {
public:
    RoadPoint(geometry::Vector2D position, int type): position(position), type(type){};
    geometry::Vector2D position;
    int type;
    int radius=2;
    geometry::AABB GetAABB() const override {return geometry::AABB(position - radius, position + radius);}
};

class RoadLine : public sf::Drawable {
public:
    RoadLine(std::vector<geometry::Vector2D> geometry, RoadType road_type,
            bool checkForCollisions=false,
            int numPoints=8);
            // num_points = how many points are returned to 
            // represent the road line
    bool canCollide(){return checkForCollisions;}
    std::vector<geometry::Vector2D> getSplineCoeffs() const;
    std::vector<geometry::Vector2D> getRoadPolyLine() const;
    void setRoadPoints(); // return an evenly spaced set of the geometry
    std::vector<std::shared_ptr<RoadPoint>> getRoadPoints(); // return an evenly spaced set of the geometry
    std::vector<geometry::Vector2D> getAllPoints() {return geometry;}
    sf::Color getColor(RoadType road_type);
    RoadType getRoadType();


private:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;
    void buildRoadLineGraphics();
    std::vector<geometry::Vector2D> geometry;
    std::vector<sf::Vertex> roadLines;
    std::vector<std::shared_ptr<RoadPoint>> roadPoints; // subsampled list of points
    bool checkForCollisions;
    RoadType road_type;
    int numPoints; 
};

}  // namespace nocturne
