#include <Network.hpp>


Network::Network() : roads() {

}

void Network::addRoad(std::vector<Vector2D> geometry, int lanes, float laneWidth) {
    roads.emplace_back(geometry, lanes, laneWidth);
}

    roads.push_back(road);
}

void Network::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    for (const Road& road : roads) {
        target.draw(road, states);
    }
}