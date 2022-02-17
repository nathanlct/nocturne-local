#include "geometry/segment.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {
namespace {

TEST(SegmentTest, AABBTest) {
  const Vector2D p = Vector2D(0.0, 2.0);
  const Vector2D q = Vector2D(1.0, 0.0);
  const Segment seg(p, q);
  const AABB aabb = seg.GetAABB();
  EXPECT_FLOAT_EQ(aabb.Center().x(), 0.5f);
  EXPECT_FLOAT_EQ(aabb.Center().y(), 1.0f);
  EXPECT_FLOAT_EQ(aabb.min().x(), 0.0f);
  EXPECT_FLOAT_EQ(aabb.min().y(), 0.0f);
  EXPECT_FLOAT_EQ(aabb.max().x(), 1.0f);
  EXPECT_FLOAT_EQ(aabb.max().y(), 2.0f);
}

TEST(SegmentTest, IntersectionTest) {
  Vector2D p1 = Vector2D(0.0, 0.0);
  Vector2D q1 = Vector2D(1.0, 1.0);
  Vector2D p2 = Vector2D(1.0, 0.0);
  Vector2D q2 = Vector2D(0.0, 1.0);
  EXPECT_TRUE(Segment(p1, q1).Intersects(Segment(p2, q2)));

  p1 = Vector2D(0.0, 0.0);
  q1 = Vector2D(0.0, 1.0);
  p2 = Vector2D(1.0, 0.0);
  q2 = Vector2D(0.0, 1.0);
  EXPECT_FALSE(Segment(p1, q1).Intersects(Segment(p2, q2)));
}

}  // namespace
}  // namespace geometry
}  // namespace nocturne
