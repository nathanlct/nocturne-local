#pragma once

#include <SFML/Graphics.hpp>
#include <initializer_list>
#include <string>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/vector_2d.h"
#include "static_object.h"
#include "utils/sf_utils.h"

namespace nocturne {

// RoadPoint should be treated as a single point.
// Add a dummy radius here for AABB.
constexpr float kRoadPointRadius = 1e-3;

// Default sampling rate for RoadPoints.
constexpr int64_t kSampleEveryN = 10;

enum class RoadType {
  kNone = 0,
  kLane = 1,
  kRoadLine = 2,
  kRoadEdge = 3,
  kStopSign = 4,
  kCrosswalk = 5,
  kSpeedBump = 6,
  kOthers = 7,
};

sf::Color RoadTypeColor(const RoadType& road_type);

class RoadPoint : public StaticObject {
 public:
  RoadPoint() = default;
  RoadPoint(const geometry::Vector2D& position,
            const geometry::Vector2D& neighbor_position, RoadType road_type)
      : StaticObject(position,
                     /*can_block_sight=*/false,
                     /*can_be_collided=*/false, /*check_collision=*/false),
        neighbor_position_(neighbor_position),
        road_type_(road_type),
        drawable_(utils::MakeCircleShape(position, 0.5,
                                         RoadTypeColor(road_type), true)) {}

  StaticObjectType Type() const override {
    return StaticObjectType::kRoadPoint;
  }
  RoadType road_type() const { return road_type_; }

  geometry::Vector2D neighbor_position() const { return neighbor_position_; }

  float Radius() const { return kRoadPointRadius; }

  geometry::ConvexPolygon BoundingPolygon() const override;

  geometry::AABB GetAABB() const override {
    return geometry::AABB(position_ - kRoadPointRadius,
                          position_ + kRoadPointRadius);
  }

 protected:
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

  // coordinates of the next point in the roadline
  const geometry::Vector2D neighbor_position_;

  const RoadType road_type_ = RoadType::kNone;
  std::unique_ptr<sf::CircleShape> drawable_;
};

// RoadLine is not an Object now.
class RoadLine : public sf::Drawable {
 public:
  RoadLine() = default;

  RoadLine(RoadType road_type,
           const std::initializer_list<geometry::Vector2D>& geometry_points,
           int64_t sample_every_n = 1, bool check_collision = false)
      : road_type_(road_type),
        geometry_points_(geometry_points),
        sample_every_n_(sample_every_n),
        check_collision_(check_collision) {
    InitRoadPoints();
    InitRoadLineGraphics();
  }

  RoadLine(RoadType road_type,
           const std::vector<geometry::Vector2D>& geometry_points,
           int64_t sample_every_n = 1, bool check_collision = false)
      : road_type_(road_type),
        geometry_points_(geometry_points),
        sample_every_n_(sample_every_n),
        check_collision_(check_collision) {
    InitRoadPoints();
    InitRoadLineGraphics();
  }

  RoadLine(RoadType road_type,
           std::vector<geometry::Vector2D>&& geometry_points,
           int64_t sample_every_n = 1, bool check_collision = false)
      : road_type_(road_type),
        geometry_points_(std::move(geometry_points)),
        sample_every_n_(sample_every_n),
        check_collision_(check_collision) {
    InitRoadPoints();
    InitRoadLineGraphics();
  }

  RoadType road_type() const { return road_type_; }

  int64_t sample_every_n() const { return sample_every_n_; }

  const std::vector<RoadPoint>& road_points() const { return road_points_; }

  const std::vector<geometry::Vector2D>& geometry_points() const {
    return geometry_points_;
  }

  bool check_collision() const { return check_collision_; }

  sf::Color Color() const;

 protected:
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

  void InitRoadPoints();
  void InitRoadLineGraphics();

  const RoadType road_type_ = RoadType::kNone;
  std::vector<geometry::Vector2D> geometry_points_;

  // Sample rate from geometry points.
  const int64_t sample_every_n_ = 1;
  std::vector<RoadPoint> road_points_;

  const bool check_collision_ = false;

  std::vector<sf::Vertex> graphic_points_;
};

inline RoadType ParseRoadType(const std::string& s) {
  if (s == "none") {
    return RoadType::kNone;
  } else if (s == "lane") {
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
    return RoadType::kOthers;
  }
}

}  // namespace nocturne
