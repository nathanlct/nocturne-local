#pragma once

#include <vector>
#include <cmath>

#include <SFML/Graphics.hpp>

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

class TrafficLight : public sf::Drawable {
public:
    TrafficLight(float x, float y, std::vector<LightState> lightStates, int currTime,
                std::vector<int> validTimes);
    LightState getLightState(int currTime);
    void updateTime(int newTime);

protected:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;
    float x;
    float y;
    int currTime;
    std::vector<LightState> lightStates;
    std::vector<int> validTimes; // used to check at which times traffic light data is actually available
};

}  // namespace nocturne
