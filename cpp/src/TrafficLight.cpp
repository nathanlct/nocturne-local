#include <TrafficLight.hpp>
#include "utils.hpp"
#include <iostream>
#include "geometry/vector_2d.h"
#include <SFML/Graphics.hpp>
using namespace std;

namespace nocturne{

TrafficLight::TrafficLight(float x, float y, std::vector<LightState> lightStates,
                           int currTime, std::vector<int> validTimes) : 
    x(x), y(y), lightStates(lightStates), currTime(currTime), validTimes(validTimes)
{
}

void TrafficLight::updateTime(int newTime){
    currTime = newTime;
}

LightState TrafficLight::getLightState(int currTime) {
    if (currTime > lightStates.size()){
        std::cerr << "Requested time that is larger than recorded Traffic Light States"
                  << std::endl;
    }
    return lightStates[currTime];
}

void TrafficLight::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    // check if the current time is actually a state for which we have info
    // If not, return unknown
    LightState lightState;
    auto result = std::find(validTimes.begin() , validTimes.end() , currTime);
    if (result != validTimes.end()){
        lightState = lightStates[result - validTimes.begin()];
    }
    else{
        lightState = LightState::unknown;
    }
    sf::Color color;
    switch(lightState) {
        case LightState::unknown : 
            color = sf::Color{102, 102, 255};
            break;
        case LightState::arrow_stop : 
            color = sf::Color::Blue;
            break;
        case LightState::arrow_caution :
            color = sf::Color::Magenta;
            break;
        case LightState::arrow_go :
            color = sf::Color::Cyan;
            break;
        case LightState::stop :
            color = sf::Color::Red;
            break;
        case LightState::caution :
            color = sf::Color::Yellow;
            break;
        case LightState::go :
            color = sf::Color::Green;
            break;
        case LightState::flashing_stop :
            color = sf::Color{255, 51, 255};
            break;
        case LightState::flashing_caution :
            color = sf::Color{255, 153, 51};
            break;
    }
    float radius = 3;
    sf::CircleShape pentagon(radius, 5);
    pentagon.setFillColor(color);
    pentagon.setPosition(utils::ToVector2f(geometry::Vector2D(x, y)));
    target.draw(pentagon, states);
}

}