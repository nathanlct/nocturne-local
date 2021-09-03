#include <Scenario.hpp>


Scenario::Scenario() : roadNetwork() {
    
}

void Scenario::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    target.draw(roadNetwork, states);
}