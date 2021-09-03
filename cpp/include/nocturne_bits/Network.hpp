#pragma once

#include "Road.hpp"
#include <vector>
#include <SFML/Graphics.hpp>

// class Vehicle;



class Network : public sf::Drawable {
public:
    Network();

    // Lane* getVehicleLane(Vehicle veh);

private:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;
    std::vector<Road> roads;
};