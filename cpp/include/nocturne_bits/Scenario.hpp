#pragma once

#include "Network.hpp"
#include "Object.hpp"
#include "Vehicle.hpp"
#include <string>
#include <SFML/Graphics.hpp>


class Scenario : public sf::Drawable {
public:
    Scenario();

    void load_from_xml(std::string path);
    void create();  // needs to insert vehicles etc from XML config

    void step(float dt);

private:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;
    Network roadNetwork;
    std::vector<Object*> roadObjects;
};