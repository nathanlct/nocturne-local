#include "view_field.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <string>

#include "Object.hpp"
#include "geometry/geometry_utils.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace {

using geometry::Vector2D;
using geometry::utils::kHalfPi;
using geometry::utils::kPi;
using testing::ElementsAre;

class MockObject : public Object {
 public:
  MockObject() = default;
  MockObject(int64_t id, float length, float width, const Vector2D& position,
             bool can_block_sight)
      : Object(id, length, width, position, /*heading=*/0.0f, /*speed=*/0.0f,
               /*goal_position=*/position, can_block_sight,
               /*can_be_collided=*/false, /*check_collision=*/false) {}

  std::string Type() const override { return "MockObject"; }
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

}  // namespace
}  // namespace nocturne
