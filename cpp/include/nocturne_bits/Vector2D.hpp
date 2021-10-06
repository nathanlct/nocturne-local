#pragma once

#include <cmath>
#include <SFML/Graphics.hpp>

const float pi = std::acos(-1);


class Vector2D {
public:
    Vector2D() : x(0), y(0) {}
    Vector2D(float x, float y) : x(x), y(y) {}

    static Vector2D fromPolar(float radius, float angle) {
        return Vector2D(radius * std::cos(angle), radius * std::sin(angle));
    }

    static int orientation(Vector2D p, Vector2D q, Vector2D r)
    {
        return (q.y - p.y) * (r.x - q.x) > (q.x - p.x) * (r.y - q.y);
    }
    
    static bool doIntersect(Vector2D p1, Vector2D q1, Vector2D p2, Vector2D q2)
    {
        return Vector2D::orientation(p1, q1, p2) != Vector2D::orientation(p1, q1, q2)
            && Vector2D::orientation(p2, q2, p1) != Vector2D::orientation(p2, q2, q1);
    }
 





    float x;
    float y;

    sf::Vector2f toVector2f(bool flipY = false) const {
        return sf::Vector2f(x, flipY ? -y : y);
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


    float norm() const {
        return std::sqrt(x * x + y * y);
    }

    float angle() const {
        return std::atan2(y, x);
    }

    void shift(float shiftX, float shiftY) {
        x += shiftX;
        y += shiftY;
    }
    void shift(const Vector2D& shiftVec) {
        shift(shiftVec.x, shiftVec.y);
    }
    void rotate(float angle) {
        float newX = x * std::cos(angle) - y * std::sin(angle);
        float newY = x * std::sin(angle) + y * std::cos(angle);
        x = newX;
        y = newY;
    }

    float dot(const Vector2D& other) const {
        return x * other.x + y * other.y;
    }
};

inline std::ostream& operator<<(std::ostream& os, const Vector2D& obj)
{
    os << "(" << obj.x << "; " << obj.y << ")";
    return os;
}