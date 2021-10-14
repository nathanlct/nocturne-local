#pragma once

#include "Object.hpp"
#include "Vehicle.hpp"
#include "Vector2D.hpp"
#include "Road.hpp"
#include "ImageMatrix.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <SFML/Graphics.hpp>
#include <stdexcept>

#include "json.hpp"
using json = nlohmann::json;

class Scenario : public sf::Drawable {
public:
    Scenario(std::string path);

    void loadScenario(std::string path);


    void addRoad(std::vector<Vector2D> geometry, int lanes, float laneWidth);

    void step(float dt);

    std::vector<Object*> getRoadObjects() const;
    std::vector<Road*> getRoads() const;

    sf::FloatRect getRoadNetworkBoundaries() const;

    ImageMatrix getCone(Object* object, float viewAngle = pi / 2.0f, float headTilt = 0.0f);

private:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

    std::string name;
    std::vector<Object*> roadObjects;
    std::vector<Road*> roads;
};