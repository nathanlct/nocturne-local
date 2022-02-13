#pragma once

#include <cmath>
#include <SFML/Graphics.hpp>


const float pi = std::acos(-1);

// Utility class for manipulating 2D vectors
class Vector2D {
public:
    float x;
    float y;

    Vector2D() : x(0), y(0) {}
    Vector2D(float x, float y) : x(x), y(y) {}

    // create a Vector2D from polar coordinates (angle in radians)
    static Vector2D fromPolar(float radius, float angle) {
        return Vector2D(radius * std::cos(angle), radius * std::sin(angle));
    }

    // check if points p, q and r are listed in counterclockwise order
    static inline bool ccw(const Vector2D& p, const Vector2D& q, const Vector2D& r)
    {
        return (q.y - p.y) * (r.x - q.x) > (q.x - p.x) * (r.y - q.y);
    }
    
    // check if lines (p1, p2) and (q1, q2) intersect (strictly)
    static bool doIntersect(const Vector2D& p1, const Vector2D& q1, const Vector2D& p2, const Vector2D& q2)
    {
        return Vector2D::ccw(p1, q1, p2) != Vector2D::ccw(p1, q1, q2)
            && Vector2D::ccw(p2, q2, p1) != Vector2D::ccw(p2, q2, q1);
    }

    // compute vector norm
    float norm() const {
        return std::sqrt(x * x + y * y);
    }

    // normalize vector in-place
    void normalize() {
        float vNorm = norm();
        x /= vNorm;
        y /= vNorm; 
    }

    // compute distance between two vectors
    float dist(const Vector2D& other) const {
        return (*this - other).norm();
    }
    static float dist(const Vector2D& v1, const Vector2D& v2) {
        return v1.dist(v2);
    }

    // get vector angle (in radians)
    float angle() const {
        return std::atan2(y, x);
    }

    // shift vector in-place
    void shift(const Vector2D& shiftVec) {
        shift(shiftVec.x, shiftVec.y);
    }
    void shift(float shiftX, float shiftY) {
        x += shiftX;
        y += shiftY;
    }

    // rotate vector in-place (angle in radians)
    void rotate(float angle) {
        float newX = x * std::cos(angle) - y * std::sin(angle);
        float newY = x * std::sin(angle) + y * std::cos(angle);
        x = newX;
        y = newY;
    }

    // compute dot product with another vector
    float dot(const Vector2D& other) const {
        return x * other.x + y * other.y;
    }

    // get normal vector (eg rotation of 90° clockwise and normalize)
    Vector2D normal() { // ie rotation of 90° clockwise + normalize
        Vector2D n(-y, x);
        n.normalize();
        return n;
    }

    // convert Vector2D to sf::Vector2f
    sf::Vector2f toVector2f(bool flipY = false) const {
        return sf::Vector2f(x, flipY ? -y : y);
    }

    // arithmetic operations
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

    // comparison operators
    friend bool operator==(const Vector2D& lhs, const Vector2D& rhs) { 
        return lhs.x == rhs.x && lhs.y == rhs.y; 
    }
    friend bool operator!=(const Vector2D& lhs, const Vector2D& rhs) { 
        return !(lhs == rhs); 
    }
};

// stream operators
inline std::ostream& operator<<(std::ostream& os, const Vector2D& obj)
{
    os << "(" << obj.x << "; " << obj.y << ")";
    return os;
}