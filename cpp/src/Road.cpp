#include <Road.hpp>


Road::Road() : geometry(
        { Vector2D(80, 50), Vector2D(300, 460), Vector2D(300, 600), Vector2D(400, 700), Vector2D(700, 300)}
    ), nLanes(3), laneWidth(20.0f), initialAngleDelta(pi / 2.0f), finalAngleDelta(pi / 2.0f),
    angles(), anglesDelta(), lanesGeometry()
{
    buildLanes();
    buildRoadGraphics();
}

void Road::buildLanes() {
    angles.clear();
    lanesGeometry.clear();
    
    float initialAngle = (geometry[1] - geometry[0]).angle();
    angles.push_back(initialAngle - initialAngleDelta);
    for (int i = 0; i < geometry.size() - 2; ++i) {
        Vector2D ptA = geometry[i];
        Vector2D ptB = geometry[i + 1];
        Vector2D ptC = geometry[i + 2];

        float angleAB = (ptB - ptA).angle();
        float angleBC = (ptC - ptB).angle();
        float angleDelta = (angleAB + pi - angleBC) / 2.0f;

        anglesDelta.push_back(angleDelta);
        angles.push_back(angleAB - angleDelta);
    }

    float finalAngle = (geometry[geometry.size() - 1] - geometry[geometry.size() - 2]).angle();
    angles.push_back(finalAngle - finalAngleDelta);

    for (int i = 0; i < geometry.size(); ++i) {
        Vector2D pt = geometry[i];
        float angle = angles[i];

        std::vector<Vector2D> points;

        float modifiedLaneWidth = (i == 0 || i == geometry.size() - 1) ? laneWidth : 
            laneWidth / std::abs(std::sin(anglesDelta[i-1]));
 
        for (int lane = 0; lane < nLanes + 1; ++lane) {
            float shift = lane * modifiedLaneWidth - nLanes * modifiedLaneWidth / 2.0f;

            float dx = shift * std::cos(angle);
            float dy = shift * std::sin(angle);

            points.push_back(Vector2D(pt.x - dx, pt.y - dy));
        }

        lanesGeometry.push_back(points);
    }
}

void Road::buildRoadGraphics() {
    // build lane quads
    laneQuads.clear();
    for (int segment = 0; segment < lanesGeometry.size() - 1; ++segment) {
        for (int lane = 0; lane < nLanes; ++lane) {
            sf::ConvexShape quad;
            quad.setPointCount(4);
            quad.setFillColor(sf::Color::Black);

            quad.setPoint(0, lanesGeometry[segment][lane].toVector2f());
            quad.setPoint(1, lanesGeometry[segment + 1][lane].toVector2f());
            quad.setPoint(2, lanesGeometry[segment + 1][lane + 1].toVector2f());
            quad.setPoint(3, lanesGeometry[segment][lane + 1].toVector2f());

            laneQuads.push_back(quad);
        }
    }

    // build road lines
    roadLines.clear();
    for (int segment = 0; segment < lanesGeometry.size() - 1; ++segment) {
        for (int lane = 0; lane < nLanes + 1; ++lane) {
            roadLines.push_back(
                sf::Vertex(lanesGeometry[segment][lane].toVector2f(), sf::Color::White)
            );
            roadLines.push_back(
                sf::Vertex(lanesGeometry[segment + 1][lane].toVector2f(), sf::Color::White)
            );
        }

sf::FloatRect Road::getBoundingBox() const {
    float minX = 1e10;
    float minY = 1e10;
    float maxX = -1e10;
    float maxY = -1e10;

    for (const auto& segment : lanesGeometry) {
        for (const auto& lane : segment) {
            minX = std::min(minX, lane.x);
            minY = std::min(minY, lane.y);
            maxX = std::max(maxX, lane.x);
            maxY = std::max(maxY, lane.y);
        }
    }

    return sf::FloatRect(minX, minY, maxX - minX, maxY - minY);
}

void Road::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    for (const sf::ConvexShape& quad : laneQuads) {
        target.draw(quad, states);
    }
    target.draw(&roadLines[0], roadLines.size(), sf::Lines, states);

    for (const std::vector<Vector2D>& line : lanesGeometry) {
        for (const Vector2D& point : line) {
            sf::CircleShape ptShape(3);
            ptShape.setOrigin(3, 3);
            ptShape.setFillColor(sf::Color::Red);
            ptShape.setPosition(point.x, point.y);
            target.draw(ptShape, states);
        }
    }
}