#include "view_field.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "geometry/geometry_utils.h"
#include "geometry/vector_2d.h"
#include "object.h"

namespace nocturne {
namespace {

using geometry::ConvexPolygon;
using geometry::Vector2D;
using geometry::utils::kHalfPi;
using geometry::utils::kPi;
using geometry::utils::kTwoPi;
using testing::ElementsAre;

class MockObject : public Object {
 public:
  MockObject() = default;
  MockObject(int64_t id, float length, float width, const Vector2D& position,
             bool can_block_sight)
      : Object(id, position, can_block_sight,
               /*can_be_collided=*/true, /*check_collision=*/true),
        length_(length),
        width_(width) {}

  float Radius() const override {
    return std::sqrt(length_ * length_ + width_ * width_) * 0.5f;
  }

  ConvexPolygon BoundingPolygon() const override {
    const geometry::Vector2D p0 =
        geometry::Vector2D(length_ * 0.5f, width_ * 0.5f) + position_;
    const geometry::Vector2D p1 =
        geometry::Vector2D(-length_ * 0.5f, width_ * 0.5f) + position_;
    const geometry::Vector2D p2 =
        geometry::Vector2D(-length_ * 0.5f, -width_ * 0.5f) + position_;
    const geometry::Vector2D p3 =
        geometry::Vector2D(length_ * 0.5f, -width_ * 0.5f) + position_;
    return ConvexPolygon({p0, p1, p2, p3});
  }

 protected:
  void draw(sf::RenderTarget& /*target*/,
            sf::RenderStates /*states*/) const override {}

  const float length_ = 0.0f;
  const float width_ = 0.0f;
};

TEST(ViewFieldTest, VisibleObjectsTest) {
  const ViewField vf(Vector2D(1.0f, 1.0f), 10.0f, kHalfPi,
                     geometry::utils::Radians(120.0f));

  const MockObject obj1(1, 2.0f, 1.0f, Vector2D(1.0f, 3.0f), true);
  const MockObject obj2(2, 2.0f, 1.0f, Vector2D(1.0f, -1.0f), true);
  const MockObject obj3(3, 2.0f, 1.0f, Vector2D(1.0f, 4.0f), true);
  const MockObject obj4(4, 1.5f, 1.0f, Vector2D(1.0f, 2.0f), false);
  const MockObject obj5(5, 2.0f, 1.0f, Vector2D(4.5f, 4.0f), true);
  auto ret = vf.VisibleObjects({&obj1, &obj2, &obj3, &obj4, &obj5});
  EXPECT_THAT(ret, ElementsAre(&obj1, &obj4, &obj5));

  const MockObject obj6(6, 10.0f, 1.0f, Vector2D(1.0f, 10.4f), true);
  ret = vf.VisibleObjects({&obj6});
  EXPECT_THAT(ret, ElementsAre(&obj6));
}

TEST(ViewFieldTest, FilterVisibleObjectsTest) {
  const ViewField vf(Vector2D(1.0f, 1.0f), 10.0f, kHalfPi,
                     geometry::utils::Radians(120.0f));

  const MockObject obj1(1, 2.0f, 1.0f, Vector2D(1.0f, 3.0f), true);
  const MockObject obj2(2, 2.0f, 1.0f, Vector2D(1.0f, -1.0f), true);
  const MockObject obj3(3, 2.0f, 1.0f, Vector2D(1.0f, 4.0f), true);
  const MockObject obj4(4, 1.5f, 1.0f, Vector2D(1.0f, 2.0f), false);
  const MockObject obj5(5, 2.0f, 1.0f, Vector2D(4.5f, 4.0f), true);
  std::vector<const Object*> objects = {&obj1, &obj2, &obj3, &obj4, &obj5};
  vf.FilterVisibleObjects(objects);
  EXPECT_THAT(objects, ElementsAre(&obj1, &obj4, &obj5));

  const MockObject obj6(6, 10.0f, 1.0f, Vector2D(1.0f, 10.4f), true);
  objects = std::vector<const Object*>{&obj6};
  vf.FilterVisibleObjects(objects);
  EXPECT_THAT(objects, ElementsAre(&obj6));
}

TEST(ViewFieldTest, PanoramicViewVisibleObjectsTest) {
  const ViewField vf(Vector2D(1.0f, 1.0f), 10.0f, kHalfPi, kTwoPi);

  const MockObject obj1(1, 2.0f, 1.0f, Vector2D(1.0f, 3.0f), true);
  const MockObject obj2(2, 2.0f, 1.0f, Vector2D(1.0f, -1.0f), true);
  const MockObject obj3(3, 2.0f, 1.0f, Vector2D(1.0f, 4.0f), true);
  const MockObject obj4(4, 1.5f, 1.0f, Vector2D(1.0f, 2.0f), false);
  const MockObject obj5(5, 2.0f, 1.0f, Vector2D(4.5f, 4.0f), true);
  auto ret = vf.VisibleObjects({&obj1, &obj2, &obj3, &obj4, &obj5});
  EXPECT_THAT(ret, ElementsAre(&obj1, &obj2, &obj4, &obj5));

  const MockObject obj6(6, 10.0f, 1.0f, Vector2D(1.0f, 10.4f), true);
  ret = vf.VisibleObjects({&obj6});
  EXPECT_THAT(ret, ElementsAre(&obj6));
}

}  // namespace
}  // namespace nocturne
