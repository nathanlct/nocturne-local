#include <Scenario.hpp>


Scenario::Scenario(std::string path) : roadNetwork(), roadObjects() {
    std::ifstream data(path);
    if (!data.is_open()) {
        throw std::invalid_argument("Scenario file couldn't be opened: " + path);
    }
    json j;
    data >> j;

    name = j["name"];

    for (const auto& obj : j["objects"]) {
        std::string type = obj["type"];
        Vector2D pos(obj["position"]["x"], obj["position"]["y"]);
        float width = obj["width"];
        float length = obj["length"];
        float heading = (float)obj["heading"] * pi / 180.0f;

        if (type == "vehicle") {
            Vehicle* veh = new Vehicle(pos, width, length, heading);
            roadObjects.push_back(veh);
        } else {
            std::cerr << "Unknown object type: " << type << std::endl;
        }
    }

    for (const auto& road : j["roads"]) {
        std::vector<Vector2D> geometry;
        for (const auto& pt : road["geometry"]) {
            geometry.emplace_back(pt["x"], pt["y"]);
        }
        
        int lanes = road["lanes"];
        float laneWidth = road["laneWidth"];

        roadNetwork.addRoad(geometry, lanes, laneWidth);
    }
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

sf::FloatRect Scenario::getBoundingBox() const {
    return roadNetwork.getBoundingBox();
}