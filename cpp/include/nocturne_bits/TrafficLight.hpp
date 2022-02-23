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
    // TODO(ev) convert lightStates and validTimes into a map to decrease chance of mistakes
    std::vector<LightState> lightStates; // traffic light state at each time in validTimes
    std::vector<int> validTimes; // list of times at which traffic lights are available
};

}  // namespace nocturne
