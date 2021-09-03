#include <Network.hpp>


Network::Network() : roads() {
    Road road;

    roads.push_back(road);
}

void Network::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    for (const Road& road : roads) {
        target.draw(road, states);
    }
}