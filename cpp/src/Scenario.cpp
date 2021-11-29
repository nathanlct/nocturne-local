#include <Scenario.hpp>


Scenario::Scenario(std::string path) : roadObjects(), vehicles(), roads(), imageTexture(nullptr) {
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

        bool occludes = obj["occludes"];
        bool collides = obj["collides"];
        bool checkForCollisions = obj["checkForCollisions"];

        Vector2D goalPos;
        if (obj.contains("goalPosition")) {
            goalPos.x = obj["goalPosition"]["x"];
            goalPos.y = obj["goalPosition"]["y"];
        }

        if (type == "vehicle") {
            Vehicle* veh = new Vehicle(pos, width, length, heading,
                                       occludes, collides, checkForCollisions,
                                       goalPos);
            auto ptr = std::shared_ptr<Vehicle>(veh);
            roadObjects.push_back(ptr);
            vehicles.push_back(ptr);
        } else if (type == "object") {
            Object* obj = new Object(pos, width, length, heading,
                                     occludes, collides, checkForCollisions,
                                     goalPos);
            roadObjects.push_back(std::shared_ptr<Object>(obj));
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
        bool hasLines = road.value("hasLines", true);

        Road* roadObject = new Road(geometry, lanes, laneWidth, hasLines);
        roads.push_back(std::shared_ptr<Road>(roadObject));
    }
}

void Scenario::createVehicle(float posX, float posY, float width, float length, float heading,
    bool occludes, bool collides, bool checkForCollisions, float goalPosX, float goalPosY)
{
    Vehicle* veh = new Vehicle(
        Vector2D(posX, posY), 
        width, 
        length, 
        heading,
        occludes, 
        collides, 
        checkForCollisions,
        Vector2D(goalPosX, goalPosY)
    );
    auto ptr = std::shared_ptr<Vehicle>(veh);
    roadObjects.push_back(ptr);
    vehicles.push_back(ptr);
}

void Scenario::step(float dt) {
    for (auto& object : roadObjects) {
        object->step(dt);
    }

    for (auto& object1 : roadObjects) {
        for (auto& object2 : roadObjects) {
            if (object1 == object2)
                continue;
            if (!object1->checkForCollisions && !object2->checkForCollisions)
                continue;
            if (!object1->collides || !object2->collides)
                continue;
            bool collided = checkForCollision(object1.get(), object2.get());
            if (collided) {
                object1->setCollided(true);
                object2->setCollided(true);
            }
        }
    }
}

bool Scenario::checkForCollision(const Object* object1, const Object* object2) {
    // note: right now objects are rectangles but this code works for any pair of convex polygons

    // first check for circles collision
    float dist = Vector2D::dist(object1->getPosition(), object2->getPosition());
    float minDist = object1->getRadius() + object2->getRadius();
    if (dist > minDist) {
        return false;
    }

    // then for exact collision

    // if the polygons don't intersect, then there exists a line, parallel
    // to a side of one of the two polygons, that entirely separate the two polygons

    // go over both polygons
    for (const Object* polygon : {object1, object2}) {
        // go over all of their sides 
        for (const std::pair<Vector2D,Vector2D>& line : polygon->getLines()) {
            // vector perpendicular to current polygon line
            const Vector2D normal = (line.second - line.first).normal(); 

            // project all corners of polygon 1 onto that line
            // min and max represent the boundaries of the polygon's projection on the line
            double min1 = std::numeric_limits<double>::max();
            double max1 = std::numeric_limits<double>::min();
            for (const Vector2D& pt : object1->getCorners()) {
                const double projected = normal.dot(pt);
                if (projected < min1) min1 = projected;
                if (projected > max1) max1 = projected;
            }

            // same for polygon 2
            double min2 = std::numeric_limits<double>::max();
            double max2 = std::numeric_limits<double>::min();
            for (const Vector2D& pt : object2->getCorners()) {
                const double projected = normal.dot(pt);
                if (projected < min2) min2 = projected;
                if (projected > max2) max2 = projected;
            }

            if (max1 < min2 || max2 < min1) {
                // we have a line separating both polygons
                return false;
            }
         }
    }

    // we didn't find any line separating both polygons
    return true;
}

void Scenario::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    for (const auto& road : roads) {
        target.draw(*road, states);
    }
    for (const auto& object : roadObjects) {
        target.draw(*object, states);
    }
}

// std::vector<std::unique_ptr<Object, py::nodelete>> Scenario::getRoadObjects() { 
std::vector<std::shared_ptr<Object>> Scenario::getRoadObjects() { 
    return roadObjects;
}

std::vector<std::shared_ptr<Vehicle>> Scenario::getVehicles() { 
    return vehicles;
}

void Scenario::removeObject(Object* object) {
    for (auto it = vehicles.begin(); it != vehicles.end(); ) {
        if ((*it).get() == object) {
            it = vehicles.erase(it);
        } else {
            it++;
        }
    }
    
    for (auto it = roadObjects.begin(); it != roadObjects.end(); ) {
        if ((*it).get() == object) {
            it = roadObjects.erase(it);
        } else {
            it++;
        }
    }
}

sf::FloatRect Scenario::getRoadNetworkBoundaries() const {
    float minX, minY, maxX, maxY;
    bool first = true;

    for (const auto& road : roads) {
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

bool Scenario::isVehicleOnRoad(const Object& object) const {
    bool found;
    for (const Vector2D& corner: object.getCorners()) {
        found = false;

        for (const std::shared_ptr<Road>& roadptr : roads) {
            const std::vector<Vector2D> roadPolygon = roadptr->getRoadPolygon();
            const size_t nQuads = roadPolygon.size() / 2 - 1;

            for (size_t quadIdx = 0; quadIdx < nQuads; quadIdx++) {
                const std::vector<Vector2D> quadCorners {
                    roadPolygon[quadIdx],
                    roadPolygon[quadIdx + 1],
                    roadPolygon[roadPolygon.size() - 1 - quadIdx - 1],
                    roadPolygon[roadPolygon.size() - 1 - quadIdx]
                };

                // test whether {corner} is within ({intersects}) with the polygon defined by {quadCorners}
                bool intersects = false;
                for (int i = 0, j = quadCorners.size() - 1; i < quadCorners.size(); j = i++) {
                    if (((quadCorners[i].y > corner.y) != (quadCorners[j].y > corner.y)) &&
                        (corner.x < (quadCorners[j].x - quadCorners[i].x) * (corner.y-quadCorners[i].y) / 
                        (quadCorners[j].y - quadCorners[i].y) + quadCorners[i].x)) 
                    {
                        intersects = !intersects;
                    }
                }

                if (intersects) {
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            return false;
        }
    }

    return true;
}

ImageMatrix Scenario::getCone(Object* object, float viewAngle, float headTilt) { // args in radians
    float circleRadius = 300.0f;
    float renderedCircleRadius = 150.0f;

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
    sf::View view(center.toVector2f(true), sf::Vector2f(2.0f * circleRadius, 2.0f * circleRadius));
    view.rotate(- object->getHeading() * 180.0f / pi + 90.0f);
    texture->setView(view);

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

        texture->draw(&outerCircle[0], outerCircle.size(), sf::TriangleFan); //, renderTransform);
    }

    // draw cone
    if (viewAngle < 2.0f * pi) {
        std::vector<sf::Vertex> innerCircle; // todo precompute just once

        innerCircle.push_back(sf::Vertex(sf::Vector2f(0.0f, 0.0f), sf::Color::Black));
        float startAngle = pi / 2.0f + headTilt + viewAngle / 2.0f;
        float endAngle = pi / 2.0f + headTilt + 2.0f * pi - viewAngle / 2.0f;

        int nPoints = 80; // todo function of angle
        for (int i = 0; i < nPoints; ++i) {
            float angle = startAngle + i * (endAngle - startAngle) / (nPoints - 1);
            Vector2D pt = Vector2D::fromPolar(r, angle);
            innerCircle.push_back(sf::Vertex(pt.toVector2f(), sf::Color::Black));
        }

        texture->draw(&innerCircle[0], innerCircle.size(), sf::TriangleFan, renderTransform);
    }

    renderTransform.rotate(- object->getHeading() * 180.0f / pi + 90.0f);

    // draw obstructions
    std::vector<std::shared_ptr<Object>> roadObjects = getRoadObjects(); // todo optimize with objects in range only (quadtree?)
    
    for (const auto& obj : roadObjects) {
        if (obj.get() != object && obj->occludes) {
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
    unsigned char* pixelsArr = (unsigned char*)img.getPixelsPtr();

    return ImageMatrix(pixelsArr, 300, 300, 4);
}   


ImageMatrix Scenario::getImage(Object* object, bool renderGoals) {
    int squareSide = 300;

    if (imageTexture == nullptr) {
        sf::ContextSettings settings;
        settings.antialiasingLevel = 1;
        imageTexture = new sf::RenderTexture();
        imageTexture->create(squareSide, squareSide, settings);
    }
    sf::RenderTexture* texture = imageTexture;

    sf::Transform renderTransform;
    renderTransform.scale(1, -1); // horizontal flip

    texture->clear(sf::Color(50, 50, 50));

    // same as in Simulation.cpp
    float padding = 50.0f;
    sf::FloatRect scenarioBounds = getRoadNetworkBoundaries();
    scenarioBounds.top = - scenarioBounds.top - scenarioBounds.height;
    scenarioBounds.top -= padding;
    scenarioBounds.left -= padding;
    scenarioBounds.width += 2 * padding;
    scenarioBounds.height += 2 * padding;
    sf::Vector2f center = sf::Vector2f(
        scenarioBounds.left + scenarioBounds.width / 2.0f, 
        scenarioBounds.top + scenarioBounds.height / 2.0f
    );
    sf::Vector2f size = sf::Vector2f(squareSide, squareSide) 
        * std::max(scenarioBounds.width / squareSide, scenarioBounds.height / squareSide);
    sf::View view(center, size);

    texture->setView(view);

    for (const auto& road : roads) {
        texture->draw(*road, renderTransform);
    }
    if (object == nullptr) {            
        for (const auto& obj : roadObjects) {
            texture->draw(*obj, renderTransform);
            
            if (renderGoals && obj->getType() == "Vehicle") {
                // draw goal destination
                float radius = 10;
                sf::CircleShape ptShape(radius);
                ptShape.setOrigin(radius, radius);
                ptShape.setFillColor(obj->color);
                ptShape.setPosition(obj->goalPosition.toVector2f());
                texture->draw(ptShape, renderTransform);
            }
        }
    } else {
        texture->draw(*object, renderTransform);

        if (renderGoals) {
            // draw goal destination
            float radius = 10;
            sf::CircleShape ptShape(radius);
            ptShape.setOrigin(radius, radius);
            ptShape.setFillColor(object->color);
            ptShape.setPosition(object->goalPosition.toVector2f());
            texture->draw(ptShape, renderTransform);
        }
    }

    // render texture and return
    texture->display();

    sf::Image img = texture->getTexture().copyToImage();
    unsigned char* pixelsArr = (unsigned char*)img.getPixelsPtr();

    return ImageMatrix(pixelsArr, squareSide, squareSide, 4);
}   