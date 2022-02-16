#include <RoadLine.hpp>
#include "utils.hpp"
#include <iostream>
using namespace std;

namespace nocturne{

RoadLine::RoadLine(std::vector<geometry::Vector2D> geometry, RoadType road_type, 
                   bool checkForCollisions, int numPoints) : 
    geometry(geometry), splineCoefficients(), road_type(road_type),
    numPoints(numPoints), checkForCollisions(checkForCollisions), roadPoints()
{
    computeSpline();
    setRoadPoints();
    setState();
    buildRoadLineGraphics();
}

void RoadLine::setState(){
    std::vector<float> state;
    for (auto& roadPoint : roadPoints){
        state.push_back(roadPoint.x());
        state.push_back(roadPoint.y());
    }
    if (geometry.size() < numPoints){
        int diff = numPoints - geometry.size();
        for (int i = 0; i < diff; i++){
            //TODO(ev) hardcoding invalid points!
            roadPoints.push_back(geometry::Vector2D(-100, -100));
        }
    }
    state.push_back(float(road_type));
}

std::vector<float> RoadLine::getState(){
    return state;
}

std::vector<float> RoadLine::getLocalState(geometry::Vector2D vehPos){
    std::vector<float> localState;
    int numRoadPoints = getState().size();
    for (int i = 0; i < numRoadPoints; i++){
        // get the local x and y coordinates
        localState.push_back(state[2 * i] - vehPos.x());
        localState.push_back(state[2 * i + 1] - vehPos.y());
    }
    localState.push_back(state.back());
    return localState;
}

void RoadLine::setRoadPoints(){
    int stepSize = geometry.size() / numPoints;
    // handle case were numPoints is greater than the 
    // size of the list
    if (numPoints > geometry.size()){
        int diff = numPoints - geometry.size();
        for (int i = 0; i < geometry.size(); i++){
            roadPoints.push_back(geometry[i]);
        }
    }
    else{
        // TODO(ev) actually we need to make sure we include
        // the most extremal points
        // consider using polyline decimation algos
        for(int i = 0; i < numPoints - 1; i++){
            roadPoints.push_back(geometry[i * stepSize]);
        }
        roadPoints.push_back(geometry.back());
    }
}

 std::vector<geometry::Vector2D> RoadLine::getRoadPoints(){
     return roadPoints;
 }

void RoadLine::computeSpline() {
    
}

RoadType RoadLine::getRoadType() {
    return road_type;
}

sf::Color RoadLine::getColor(RoadType road_type) {
    // TODO(ev) handle striped and so on
    switch(road_type) {
        case RoadType::lane : 
            return sf::Color::Yellow;
        case RoadType::road_line : 
            return sf::Color::Blue;
        case RoadType::road_edge :
            return sf::Color::Green;
        case RoadType::stop_sign :
            return sf::Color::Red;
        case RoadType::crosswalk :
            return sf::Color::Magenta;
        case RoadType::speed_bump :
            return sf::Color::Cyan;
        default: 
            return sf::Color::Transparent;
    }
}

void RoadLine::buildRoadLineGraphics() {

    // build road lines
    roadLines.clear();

    for (int segment = 0; segment < geometry.size(); ++segment) {
        roadLines.push_back(sf::Vertex(utils::ToVector2f(geometry[segment]), getColor(getRoadType())));
    }
    // this is just for debugging
    // for (int segment = 0; segment < roadPoints.size(); ++segment) {
    //     roadLines.push_back(sf::Vertex(roadPoints[segment].toVector2f(), getColor(getRoadType())));
    // }
}

void RoadLine::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    target.draw(&roadLines[0], roadLines.size(), sf::LineStrip, states);
}

}