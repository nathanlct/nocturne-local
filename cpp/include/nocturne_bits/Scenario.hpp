#pragma once

#include "Network.hpp"
#include "Object.hpp"
#include "Vehicle.hpp"
#include "Vector2D.hpp"
#include <string>
#include <fstream>
#include <SFML/Graphics.hpp>
#include <stdexcept>

#include "json.hpp"
using json = nlohmann::json;

class Scenario : public sf::Drawable {
public:
    Scenario(std::string path);

    void step(float dt);
    sf::FloatRect getBoundingBox() const;

private:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

    std::string name;
    Network roadNetwork;
    std::vector<Object*> roadObjects;
};