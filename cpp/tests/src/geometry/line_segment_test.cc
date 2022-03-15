#include "geometry/line_segment.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cmath>

#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {
namespace {

constexpr float kTestEps = 1e-5;

TEST(LineSegmentTest, AABBTest) {
  const Vector2D p(0.0f, 2.0f);
  const Vector2D q(1.0f, 0.0f);
  const LineSegment seg(p, q);
  const AABB aabb = seg.GetAABB();
  EXPECT_FLOAT_EQ(aabb.min().x(), 0.0f);
  EXPECT_FLOAT_EQ(aabb.min().y(), 0.0f);
  EXPECT_FLOAT_EQ(aabb.max().x(), 1.0f);
  EXPECT_FLOAT_EQ(aabb.max().y(), 2.0f);
  EXPECT_FLOAT_EQ(aabb.Center().x(), 0.5f);
  EXPECT_FLOAT_EQ(aabb.Center().y(), 1.0f);
}

TEST(LineSegmentTest, NormalVectorTest) {
  Vector2D p(1.0f, 1.0f);
  Vector2D q(2.0f, 1.0f);
  Vector2D normal_vector = LineSegment(p, q).NormalVector();
  EXPECT_FLOAT_EQ(normal_vector.x(), 0.0f);
  EXPECT_FLOAT_EQ(normal_vector.y(), 1.0f);

  p = Vector2D(-1.0f, -1.0f);
  q = Vector2D(-2.0f, -2.0f);
  normal_vector = LineSegment(p, q).NormalVector();
  EXPECT_FLOAT_EQ(normal_vector.x(), M_SQRT1_2);
  EXPECT_FLOAT_EQ(normal_vector.y(), -M_SQRT1_2);
}

TEST(LineSegmentTest, ContainsTest) {
  const Vector2D p(0.0, 0.0);
  const Vector2D q(1.0, 1.0);
  const LineSegment seg(p, q);
  EXPECT_TRUE(seg.Contains(p));
  EXPECT_TRUE(seg.Contains(q));
  EXPECT_TRUE(seg.Contains(p + kTestEps));
  EXPECT_TRUE(seg.Contains(q - kTestEps));
  EXPECT_FALSE(seg.Contains(Vector2D(0.5f + kTestEps, 0.5f - kTestEps)));
  EXPECT_FALSE(seg.Contains(Vector2D(0.0f - kTestEps, 0.0f)));
  EXPECT_FALSE(seg.Contains(Vector2D(1.0f, 1.0f + kTestEps)));
}

TEST(LineSegmentTest, IntersectsTest) {
  Vector2D p1(0.0, 0.0);
  Vector2D q1(1.0, 1.0);
  Vector2D p2(1.0, 0.0);
  Vector2D q2(0.0, 1.0);
  EXPECT_TRUE(LineSegment(p1, q1).Intersects(LineSegment(p2, q2)));

  p1 = Vector2D(0.0, 0.0);
  q1 = Vector2D(0.0, 1.0);
  p2 = Vector2D(1.0, 0.0);
  q2 = Vector2D(0.0, 1.0);
  EXPECT_FALSE(LineSegment(p1, q1).Intersects(LineSegment(p2, q2)));
}

}  // namespace
}  // namespace geometry
}  // namespace nocturne
