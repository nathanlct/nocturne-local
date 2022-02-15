#include "geometry/segment.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {
namespace {

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
