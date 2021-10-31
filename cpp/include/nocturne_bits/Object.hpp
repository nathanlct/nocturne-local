#pragma once

#include <vector>
#include <iostream>
#include <string>
#include "Vector2D.hpp"
#include <SFML/Graphics.hpp>

class Point;

class Object : public sf::Drawable {
public:
    Object(Vector2D position, float width, float length, float heading,
           bool occludes, bool collides, bool checkForCollisions,
           Vector2D goalPosition);

    // bool intersectsWith(Object* other) const; // fast spherical pre-check, then accurate rectangular check
    // std::vector<Point> getCorners() const;

    // void move(); // move according to pos, heading and speed
    virtual void step(float dt);

    Vector2D getPosition() const;
    Vector2D getGoalPosition() const;
    float getSpeed() const;
    float getHeading() const;
    float getWidth() const;
    float getLength() const;
    int getID() const;
    std::string getType() const;
    float getRadius() const;  // radius of the minimal circle of center {position} that includes the whole polygon
    std::vector<Vector2D> getCorners() const;
    std::vector<std::pair<Vector2D,Vector2D>> getLines() const;

    void setCollided(bool collided);
    bool getCollided() const;

    sf::RenderTexture* coneTexture;
    sf::RenderTexture* goalTexture;

    static int nextID;

protected:
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const;

    // bool solid;

    Vector2D position;
    float width;
    float length;
    float heading;

    float speed;
    int id;
    std::string type;

    bool hasCollided;

public: // tmp
    bool occludes;
    bool collides;
    bool checkForCollisions;

    Vector2D goalPosition;
};