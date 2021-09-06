#include <Network.hpp>


Network::Network() : roads() {

}

void Network::addRoad(std::vector<Vector2D> geometry, int lanes, float laneWidth) {
    roads.emplace_back(geometry, lanes, laneWidth);
}

sf::FloatRect Network::getBoundingBox() const {
    float minX = 1e10;
    float minY = 1e10;
    float maxX = -1e10;
    float maxY = -1e10;

    for (const auto& road : roads) {
        const sf::FloatRect bounds = road.getBoundingBox();
        minX = std::min(minX, bounds.left);
        minY = std::min(minY, bounds.top);
        maxX = std::max(maxY, bounds.left + bounds.width);
        maxY = std::max(maxY, bounds.top + bounds.height);
    }

    return sf::FloatRect(minX, minY, maxX - minX, maxY - minY);
}

void Network::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    for (const Road& road : roads) {
        target.draw(road, states);
    }
}