#include <Scenario.hpp>


Scenario::Scenario() : roadNetwork(), roadObjects() {
    Vehicle* veh = new Vehicle();
    roadObjects.push_back(veh);
}

void Scenario::step(float dt) {
    for (auto* object : roadObjects) {
        object->step(dt);
    }
}

void Scenario::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    target.draw(roadNetwork, states);
    for (const Object* object : roadObjects) {
        target.draw(*object, states);
    }
}