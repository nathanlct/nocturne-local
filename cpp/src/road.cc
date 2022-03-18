#include "road.h"

#include "geometry/vector_2d.h"
#include "utils/sf_utils.h"

namespace nocturne {

geometry::ConvexPolygon RoadPoint::BoundingPolygon() const {
  const geometry::Vector2D p0 =
      position_ + geometry::Vector2D(kRoadPointRadius, kRoadPointRadius);
  const geometry::Vector2D p1 =
      position_ + geometry::Vector2D(-kRoadPointRadius, kRoadPointRadius);
  const geometry::Vector2D p2 =
      position_ + geometry::Vector2D(-kRoadPointRadius, -kRoadPointRadius);
  const geometry::Vector2D p3 =
      position_ + geometry::Vector2D(kRoadPointRadius, -kRoadPointRadius);
  return geometry::ConvexPolygon({p0, p1, p2, p3});
}

sf::Color RoadLine::Color() const {
  switch (road_type_) {
    case RoadType::kLane: {
      return sf::Color::Yellow;
    }
    case RoadType::kRoadLine: {
      return sf::Color::Blue;
    }
    case RoadType::kRoadEdge: {
      return sf::Color::Green;
    }
    case RoadType::kStopSign: {
      return sf::Color::Red;
    }
    case RoadType::kCrosswalk: {
      return sf::Color::Magenta;
    }
    case RoadType::kSpeedBump: {
      return sf::Color::Cyan;
    }
    default: {
      return sf::Color::Transparent;
    }
  };
}

void RoadLine::draw(sf::RenderTarget& target, sf::RenderStates states) const {
  target.draw(graphic_points_.data(), graphic_points_.size(), sf::LineStrip,
              states);
}

void RoadLine::InitRoadPoints() {
  road_points_.reserve(num_road_points_);
  const int64_t n = geometry_points_.size();
  if (n < num_road_points_) {
    for (int64_t i = 0; i < n; ++i) {
      road_points_.emplace_back(i, geometry_points_[i], road_type_);
    }
    // Padding RoadPoints.
    for (int64_t i = n; i < num_road_points_; ++i) {
      road_points_.emplace_back(i, geometry::Vector2D(0.0f, 0.0f),
                                RoadType::kNone);
    }
  } else {
    const int64_t step = n / num_road_points_;
    for (int64_t i = 0; i < num_road_points_ - 1; ++i) {
      road_points_.emplace_back(i, geometry_points_[i * step], road_type_);
    }
    road_points_.emplace_back(num_road_points_ - 1, geometry_points_.back(),
                              road_type_);
  }
}

void RoadLine::InitRoadLineGraphics() {
  const int64_t n = geometry_points_.size();
  graphic_points_.reserve(n);
  for (const geometry::Vector2D& p : geometry_points_) {
    graphic_points_.emplace_back(sf::Vertex(utils::ToVector2f(p), Color()));
  }
}

RoadType ParseRoadType(const std::string& s) {
  if (s == "lane") {
    return RoadType::kLane;
  } else if (s == "road_line") {
    return RoadType::kRoadLine;
  } else if (s == "road_edge") {
    return RoadType::kRoadEdge;
  } else if (s == "stop_sign") {
    return RoadType::kStopSign;
  } else if (s == "crosswalk") {
    return RoadType::kCrosswalk;
  } else if (s == "speed_bump") {
    return RoadType::kSpeedBump;
  } else {
    return RoadType::kNone;
  }
}

}  // namespace nocturne
