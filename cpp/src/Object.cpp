#include <Object.hpp>


Object::Object() :
    position(200, 200), width(10), length(20), heading(1.2),
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
}