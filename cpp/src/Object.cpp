#include <Object.hpp>


Object::Object(Vector2D position, float width, float length, float heading) :
    position(position), width(width), length(length), heading(heading),
    speed(0)
{

}

void Object::step(float dt) {

}

void Object::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    sf::RectangleShape rect(sf::Vector2f(length, width));
    rect.setOrigin(length / 2.0f, width / 2.0f);
    rect.setPosition(position.toVector2f());
    rect.setRotation(heading * 180 / pi);
    rect.setFillColor(sf::Color::Green);
    target.draw(rect, states);

    sf::CircleShape circle(3);
    circle.setOrigin(3, 3);
    circle.setFillColor(sf::Color::Red);
    circle.setPosition(position.toVector2f());
    target.draw(circle, states);

    for (Vector2D corner : getCorners()) {
        sf::CircleShape circle(2);
        circle.setOrigin(2, 2);
        circle.setFillColor(sf::Color::Red);
        circle.setPosition(corner.toVector2f());
        target.draw(circle, states);
    }
}

Vector2D Object::getPosition() const {
    return position;
}
float Object::getHeading() const {
    return heading;
}
float Object::getWidth() const {
    return width;
}
float Object::getLength() const {
    return length;
}
std::vector<Vector2D> Object::getCorners() const {
    std::vector<Vector2D> corners;
    // create points
    for (auto [multX, multY] : (std::vector<std::pair<int,int>>){ {1, 1}, {1, -1}, {-1, -1}, {-1, 1} }) {
        corners.push_back(Vector2D(multX * length / 2.0f, multY * width / 2.0f));
    }
    // rotate points
    for (Vector2D& pt : corners) {
        pt.rotate(heading);
    }
    // translate points
    for (Vector2D& pt : corners) {
        pt.shift(position);
    }
    return corners;
}

std::vector<std::pair<Vector2D,Vector2D>> Object::getLines() const {
    std::vector<Vector2D> corners = getCorners();
    std::vector<std::pair<Vector2D,Vector2D>> lines;
    for (int i = 0; i < corners.size(); ++i) {
        lines.emplace_back(corners[i], corners[(i+1)%corners.size()]);
    }
    return lines;
}