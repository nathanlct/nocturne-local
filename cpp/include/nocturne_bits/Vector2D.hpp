#pragma once

#include <cmath>
#include <SFML/Graphics.hpp>

const float pi = std::acos(-1);


class Vector2D {
public:
    Vector2D() : x(0), y(0) {}
    Vector2D(float x, float y) : x(x), y(y) {}

    float x;
    float y;

    sf::Vector2f toVector2f() const {
        return sf::Vector2f(x, y);
    }


    Vector2D& operator+=(const Vector2D& rhs) {
        x += rhs.x; y += rhs.y; return *this;
    }
    friend Vector2D operator+(Vector2D lhs, const Vector2D& rhs) {
        lhs += rhs; return lhs;
    }

    Vector2D& operator-=(const Vector2D& rhs) {
        x -= rhs.x; y -= rhs.y; return *this;
    }
    friend Vector2D operator-(Vector2D lhs, const Vector2D& rhs) {
        lhs -= rhs; return lhs;
    }

    Vector2D& operator*=(const Vector2D& rhs) {
        x *= rhs.x; y *= rhs.y; return *this;
    }
    friend Vector2D operator*(Vector2D lhs, const Vector2D& rhs) {
        lhs *= rhs; return lhs;
    }

    Vector2D& operator*=(float rhs) {
        x *= rhs; y *= rhs; return *this;
    }
    friend Vector2D operator*(Vector2D lhs, float rhs) {
        lhs *= rhs; return lhs;
    }

    Vector2D& operator/=(const Vector2D& rhs) {
        x /= rhs.x; y /= rhs.y; return *this;
    }
    friend Vector2D operator/(Vector2D lhs, const Vector2D& rhs) {
        lhs /= rhs; return lhs;
    }

    Vector2D& operator/=(float rhs) {
        x /= rhs; y /= rhs; return *this;
    }
    friend Vector2D operator/(Vector2D lhs, float rhs) {
        lhs /= rhs; return lhs;
    }

    friend bool operator==(const Vector2D& lhs, const Vector2D& rhs){ 
        return lhs.x == rhs.x && lhs.y == rhs.y; 
    }
    friend bool operator!=(const Vector2D& lhs, const Vector2D& rhs){ return !(lhs == rhs); }


    float norm() {
        return std::sqrt(x * x + y * y);
    }

    float angle() {
        return std::atan2(y, x);
    }
};