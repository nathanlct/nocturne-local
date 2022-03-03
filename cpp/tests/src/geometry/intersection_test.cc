#include "geometry/intersection.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "geometry/line_segment.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {
namespace {

constexpr float kEps = 1e-5;

TEST(IntersectsTest, AABBSegmentTest) {
  const AABB aabb(0.0f, 0.0f, 4.0f, 2.0f);

  const LineSegment seg1(Vector2D(1.0f, 1.0f), Vector2D(1.5f, 1.5f));
  EXPECT_TRUE(Intersects(aabb, seg1));
  EXPECT_TRUE(Intersects(seg1, aabb));
  const LineSegment seg2(Vector2D(-1.0f, -1.0f), Vector2D(0.0f, 0.0f));
  EXPECT_TRUE(Intersects(aabb, seg2));
  EXPECT_TRUE(Intersects(seg2, aabb));
  const LineSegment seg3(Vector2D(-1.0f, 1.0f), Vector2D(2.0f, -1.0f));
  EXPECT_TRUE(Intersects(aabb, seg3));
  EXPECT_TRUE(Intersects(seg3, aabb));
  const LineSegment seg4(Vector2D(1.0f, 3.0f), Vector2D(1.0f, 1.0f));
  EXPECT_TRUE(Intersects(aabb, seg4));
  EXPECT_TRUE(Intersects(seg4, aabb));

  const LineSegment seg5(Vector2D(4.0f + kEps, 0.5f),
                         Vector2D(4.0f + kEps, 1.0f));
  EXPECT_FALSE(Intersects(aabb, seg5));
  EXPECT_FALSE(Intersects(seg5, aabb));
  const LineSegment seg6(Vector2D(3.0f + kEps, -1.0f),
                         Vector2D(5.0f + kEps, 1.0f));
  EXPECT_FALSE(Intersects(aabb, seg6));
  EXPECT_FALSE(Intersects(seg6, aabb));
}

TEST(IntersectsTest, ConvexPolygonSegmentTest) {
  const ConvexPolygon polygon({Vector2D(1.0f, 0.0f), Vector2D(0.0f, 1.0f),
                               Vector2D(-1.0f, 0.0f), Vector2D(0.0f, -1.0f)});

  const LineSegment seg1(Vector2D(0.0f, 0.5f), Vector2D(0.0f, -0.5f));
  EXPECT_TRUE(Intersects(polygon, seg1));
  EXPECT_TRUE(Intersects(seg1, polygon));
  const LineSegment seg2(Vector2D(-0.5f, -0.5f), Vector2D(-0.5f, -1.0f));
  EXPECT_TRUE(Intersects(polygon, seg2));
  EXPECT_TRUE(Intersects(seg2, polygon));
  const LineSegment seg3(Vector2D(-1.0f, 0.5f), Vector2D(1.0f, 1.0f));
  EXPECT_TRUE(Intersects(polygon, seg3));
  EXPECT_TRUE(Intersects(seg3, polygon));
  const LineSegment seg4(Vector2D(1.0f, 1.0f), Vector2D(-1.0f, -1.0f));
  EXPECT_TRUE(Intersects(polygon, seg4));
  EXPECT_TRUE(Intersects(seg4, polygon));

  const LineSegment seg5(Vector2D(-1.0f, -1.0f - kEps),
                         Vector2D(1.0f, -1.0f - kEps));
  EXPECT_FALSE(Intersects(polygon, seg5));
  EXPECT_FALSE(Intersects(seg5, polygon));
  const LineSegment seg6(Vector2D(-3.0f, 0.5f), Vector2D(-2.0f, 1.0f));
  EXPECT_FALSE(Intersects(polygon, seg6));
  EXPECT_FALSE(Intersects(seg6, polygon));
}

}  // namespace
}  // namespace geometry
}  // namespace nocturne
