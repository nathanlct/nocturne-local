#pragma once

#include <vector>
#include <cmath>
#include <memory>

#include <SFML/Graphics.hpp>
#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"

namespace nocturne {

enum LightState {
    unknown,
    arrow_stop,
    arrow_caution,
    arrow_go,
    stop,
    caution,
    go,
    flashing_stop,
    flashing_caution,
};

// AABB class used to store the traffic lights for visibility checking
class TrafficLightBox : public geometry::AABBInterface {
public:
    TrafficLightBox(geometry::Vector2D position, int tlIndex): position(position), tlIndex(tlIndex){};
    geometry::Vector2D position;
    int tlIndex;
    int radius=2;
    geometry::AABB GetAABB() const override {return geometry::AABB(position - radius, position + radius);}
};

class TrafficLight : public sf::Drawable {
public:
    TrafficLight(float x, float y, std::vector<LightState> lightStates, int currTime,
                std::vector<int> validTimes);
    int getLightState();
    geometry::Vector2D getPosition();
    void updateTime(int newTime);

protected:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;
    float x;
    float y;
    int currTime;
    // TODO(ev) convert lightStates and validTimes into a map to decrease chance of mistakes
    std::vector<LightState> lightStates; // traffic light state at each time in validTimes
    std::vector<int> validTimes; // list of times at which traffic lights are available
};

}  // namespace nocturne
