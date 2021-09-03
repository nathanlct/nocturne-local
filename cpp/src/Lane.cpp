#include <Lane.hpp>

#include <cmath>

Lane::Lane(std::vector<Vector2D> geometry, float width) :
    geometry(geometry),
    width(width)
{}

void Lane::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    const float pi = std::acos(-1);

    for (int i = 0; i < geometry.size() - 1; ++i) {
        Vector2D from = geometry[i];
        Vector2D to = geometry[i + 1];

        float rWidth = width;
        Vector2D rVec = (to - from);
        float rLength = rVec.norm();
        float rAngle = rVec.angle();
        Vector2D center = (from + to) / 2.0;

        std::cout << "from (" << from.x << ";" << from.y << ")" << " to (" << to.x << ";" << to.y << "), angle: " << rAngle*180/pi << std::endl;


        // sf::RectangleShape line(sf::Vector2f(rLength, rWidth));
        // line.setRotation((to - from).angle() * 180 / 3.1415);
        // line.setFillColor(sf::Color::White);
        // line.setOrigin(rLength / 2.0, rWidth / 2.0);
        // line.setPosition((from.x + to.x) / 2.0, 800 - (from.y + to.y) / 2.0);
        // target.draw(line, states);

        for (const Vector2D& pt : { from, to, center }) {
            sf::CircleShape ptShape(3);
            ptShape.setOrigin(3, 3);
            ptShape.setFillColor(sf::Color::Red);
            ptShape.setPosition(pt.x, pt.y);
            target.draw(ptShape, states);
        }
        

        std::vector<Vector2D> corners;
        
        if (i == 0) {
            corners.push_back(Vector2D(
                from.x + rWidth / 2.0f * std::cos(rAngle - pi / 2.0f),
                from.y + rWidth / 2.0f * std::sin(rAngle - pi / 2.0f)
            ));
            corners.push_back(Vector2D(
                from.x - rWidth / 2.0f * std::cos(rAngle - pi / 2.0f),
                from.y - rWidth / 2.0f * std::sin(rAngle - pi / 2.0f)
            ));
        }

        if (i < geometry.size() - 2) {
            Vector2D nextFrom = geometry[i+1];
            Vector2D nextTo = geometry[i+2];
            float nextAngle = (nextTo - nextFrom).angle();

            float totalAngle = (rAngle + (pi - nextAngle)) / 2;

            corners.push_back(Vector2D(
                to.x + rWidth / 2.0f * std::cos(rAngle - totalAngle),
                to.y + rWidth / 2.0f * std::sin(rAngle - totalAngle)
            ));
            corners.push_back(Vector2D(
                to.x - rWidth / 2.0f * std::cos(rAngle - totalAngle),
                to.y - rWidth / 2.0f * std::sin(rAngle - totalAngle)
            ));

        } else {        
            corners.push_back(Vector2D(
                to.x + rWidth / 2.0f * std::cos(rAngle - pi / 2.0f),
                to.y + rWidth / 2.0f * std::sin(rAngle - pi / 2.0f)
            ));
            corners.push_back(Vector2D(
                to.x - rWidth / 2.0f * std::cos(rAngle - pi / 2.0f),
                to.y - rWidth / 2.0f * std::sin(rAngle - pi / 2.0f)
            ));
        }


        for (const Vector2D& pt : corners) {
            sf::CircleShape ptShape(3);
            ptShape.setOrigin(3, 3);
            ptShape.setFillColor(sf::Color::Green);
            ptShape.setPosition(pt.x, pt.y);
            target.draw(ptShape, states);
        }
        




    }
}