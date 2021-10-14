#include <Scenario.hpp>


Scenario::Scenario(std::string path) : roadObjects(), roads() {
    if (path.size() > 0) {
        loadScenario(path);
    } else {        
        throw std::invalid_argument("No scenario file inputted.");
        // TODO(nl) right now an empty scenario crashes, expectedly
        std::cout << "No scenario path inputted. Defaulting to an empty scenario." << std::endl;
    }
}

void Scenario::loadScenario(std::string path) {
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
        } else if (type == "object") {
            Object* obj = new Object(pos, width, length, heading);
            roadObjects.push_back(obj);
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

        addRoad(geometry, lanes, laneWidth);
    }
}

void Scenario::step(float dt) {
    for (auto* object : roadObjects) {
        object->step(dt);
    }
}

void Scenario::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    for (const Road* road : roads) {
        target.draw(*road, states);
    }
    for (const Object* object : roadObjects) {
        target.draw(*object, states);
    }
}

std::vector<Object*> Scenario::getRoadObjects() const { 
    return roadObjects; 
}

void Scenario::addRoad(std::vector<Vector2D> geometry, int lanes, float laneWidth) {
    roads.push_back(new Road(geometry, lanes, laneWidth));
}

sf::FloatRect Scenario::getRoadNetworkBoundaries() const {
    float minX, minY, maxX, maxY;
    bool first = true;

    for (const auto* road : roads) {
        for (const auto& point : road->getRoadPolygon()) {
            if (first) {
                minX = maxX = point.x;
                minY = maxY = point.y;
                first = false;
            } else {
                minX = std::min(minX, point.x);
                maxX = std::max(maxX, point.x);
                minY = std::min(minY, point.y);
                maxY = std::max(maxY, point.y);
            }
        }
    }
    
    sf::FloatRect roadNetworkBounds(minX, minY, maxX - minX, maxY - minY);
    return roadNetworkBounds;
}


ImageMatrix Scenario::getCone(Object* object, float viewAngle, float headTilt) { // args in radians
    float circleRadius = 400.0f;
    float renderedCircleRadius = 200.0f;

    if (object->coneTexture == nullptr) {
        sf::ContextSettings settings;
        settings.antialiasingLevel = 1;

        object->coneTexture = new sf::RenderTexture();
        object->coneTexture->create(2.0f * renderedCircleRadius, 2.0f * renderedCircleRadius, settings);
    }
    sf::RenderTexture* texture = object->coneTexture;

    sf::Transform renderTransform;
    renderTransform.scale(1, -1); // horizontal flip

    Vector2D center = object->getPosition();
    float heading = object->getHeading() + headTilt;

    texture->clear(sf::Color(50, 50, 50));
    texture->setView(sf::View(center.toVector2f(true), sf::Vector2f(2.0f * circleRadius, 2.0f * circleRadius)));

    texture->draw(*this, renderTransform); // todo optimize with objects in range only (quadtree?)

    texture->setView(sf::View(sf::Vector2f(0.0f, 0.0f), sf::Vector2f(texture->getSize())));

    // draw circle
    float r = renderedCircleRadius;
    float diag = std::sqrt(2 * r * r);

    for (int quadrant = 0; quadrant < 4; ++quadrant) {
        std::vector<sf::Vertex> outerCircle; // todo precompute just once

        float angleShift = quadrant * pi / 2.0f;

        Vector2D corner = Vector2D::fromPolar(diag, pi / 4.0f + angleShift);
        outerCircle.push_back(sf::Vertex(corner.toVector2f(), sf::Color::Black));

        int nPoints = 20;
        for (int i = 0; i < nPoints; ++i) {
            float angle = angleShift + i * (pi / 2.0f) / (nPoints - 1);

            Vector2D pt = Vector2D::fromPolar(r, angle);
            outerCircle.push_back(sf::Vertex(pt.toVector2f(), sf::Color::Black));
        }

        texture->draw(&outerCircle[0], outerCircle.size(), sf::TriangleFan, renderTransform);
    }

    // draw cone
    if (viewAngle < 2.0f * pi) {
        std::vector<sf::Vertex> innerCircle; // todo precompute just once

        innerCircle.push_back(sf::Vertex(sf::Vector2f(0.0f, 0.0f), sf::Color::Black));

        float startAngle = heading + viewAngle / 2.0f;
        float endAngle = heading + 2.0f * pi - viewAngle / 2.0f;

        int nPoints = 80; // todo function of angle
        for (int i = 0; i < nPoints; ++i) {
            float angle = startAngle + i * (endAngle - startAngle) / (nPoints - 1);
            Vector2D pt = Vector2D::fromPolar(r, angle);
            innerCircle.push_back(sf::Vertex(pt.toVector2f(), sf::Color::Black));
        }

        texture->draw(&innerCircle[0], innerCircle.size(), sf::TriangleFan, renderTransform);
    }

    // draw obstructions
    std::vector<Object*> roadObjects = getRoadObjects(); // todo optimize with objects in range only (quadtree?)
    
    for (const Object* obj : roadObjects) {
        if (obj != object) {
            auto lines = obj->getLines();
            for (const auto& [pt1, pt2] : lines) {

                int nIntersections = 0;
                for (const auto& [pt3, pt4] : lines) {
                    if (pt1 != pt3 && pt1 != pt4 && Vector2D::doIntersect(pt1, center, pt3, pt4)) {
                        nIntersections++;
                        break;
                    }
                }
                for (const auto& [pt3, pt4] : lines) {
                    if (pt2 != pt3 && pt2 != pt4 && Vector2D::doIntersect(pt2, center, pt3, pt4)) {
                        nIntersections++;
                        break;
                    }
                }

                if (nIntersections >= 1) {
                    sf::ConvexShape hiddenArea;


                    float angle1 = (pt1 - center).angle();
                    float angle2 = (pt2 - center).angle();
                    while (angle2 > angle1) angle2 -= 2.0f * pi;

                    int nPoints = 80; // todo function of angle
                    hiddenArea.setPointCount(nPoints + 2);

                    hiddenArea.setPoint(0, ((pt1 - center) * 0.5f).toVector2f());
                    for (int i = 0; i < nPoints; ++i) {
                        float angle = angle1 + i * (angle2 - angle1) / (nPoints - 1);
                        Vector2D pt = Vector2D::fromPolar(r, angle);
                        hiddenArea.setPoint(1 + i, pt.toVector2f());
                    }
                    hiddenArea.setPoint(nPoints + 1, ((pt2 - center) * 0.5f).toVector2f());

                    hiddenArea.setFillColor(sf::Color::Black);

                    texture->draw(hiddenArea, renderTransform);

                }
            }
        }
    }

    texture->display();

    sf::Image img = texture->getTexture().copyToImage();
    img.saveToFile('save_cpp.png');
    unsigned char* pixelsArr = (unsigned char*)img.getPixelsPtr();

    return ImageMatrix(pixelsArr, 400, 400, 4);
}   