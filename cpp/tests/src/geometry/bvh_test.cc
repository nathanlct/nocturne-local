#include "geometry/bvh.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {
namespace {

using testing::UnorderedElementsAre;

class MockObject : public AABBInterface {
 public:
  MockObject(const Vector2D& center, float radius)
      : center_(center), radius_(radius) {}

  AABB GetAABB() const override {
    return AABB(center_ - radius_, center_ + radius_);
  }

 protected:
  Vector2D center_;
  float radius_;
};

class TestBVH : public BVH {
 public:
  TestBVH(const std::vector<const AABBInterface*>& objects) : BVH(objects) {}

  int64_t MaxDepth() const {
    int64_t max_depth = 0;
    MaxDepthImpl(root_, /*cur_depth=*/1, max_depth);
    return max_depth;
  }

 protected:
  void MaxDepthImpl(const Node* cur, int64_t cur_depth,
                    int64_t& max_depth) const {
    if (cur->IsLeaf()) {
      max_depth = std::max(max_depth, cur_depth);
      return;
    }
    MaxDepthImpl(cur->LChild(), cur_depth + 1, max_depth);
    MaxDepthImpl(cur->RChild(), cur_depth + 1, max_depth);
  }
};

TEST(BVHTest, InitHierarchyPerfTest) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

  const int64_t n = 1000;
  std::vector<MockObject> objects;
  objects.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    const float x = dis(gen);
    const float y = dis(gen);
    objects.emplace_back(Vector2D(x, y), /*radius=*/1.0f);
  }
  std::vector<const AABBInterface*> objects_ptr;
  objects_ptr.reserve(n);
  for (const auto& obj : objects) {
    objects_ptr.push_back(dynamic_cast<const AABBInterface*>(&obj));
  }

  TestBVH bvh(objects_ptr);
  EXPECT_LT(bvh.MaxDepth(), 30);
}

TEST(BVHTest, CollisionCandidatesTest) {
  const MockObject obj1(Vector2D(0.0f, 0.0f), 1.0f);
  const MockObject obj2(Vector2D(0.5f, 0.5f), 1.0f);
  const MockObject obj3(Vector2D(10.0f, 0.0f), 1.0f);
  const MockObject obj4(Vector2D(10.0f, 1.5f), 1.0f);
  const MockObject obj5(Vector2D(-10.0f, -10.0f), 1.0f);

  std::vector<const AABBInterface*> objects = {
      dynamic_cast<const AABBInterface*>(&obj1),
      dynamic_cast<const AABBInterface*>(&obj2),
      dynamic_cast<const AABBInterface*>(&obj3),
      dynamic_cast<const AABBInterface*>(&obj4),
      dynamic_cast<const AABBInterface*>(&obj5)};

  TestBVH bvh(objects);
  std::vector<const AABBInterface*> candidates =
      bvh.CollisionCandidates(dynamic_cast<const AABBInterface*>(&obj1));
  EXPECT_THAT(candidates, UnorderedElementsAre(&obj1, &obj2));
  candidates =
      bvh.CollisionCandidates(dynamic_cast<const AABBInterface*>(&obj2));
  EXPECT_THAT(candidates, UnorderedElementsAre(&obj1, &obj2));
  candidates =
      bvh.CollisionCandidates(dynamic_cast<const AABBInterface*>(&obj3));
  EXPECT_THAT(candidates, UnorderedElementsAre(&obj3, &obj4));
  candidates =
      bvh.CollisionCandidates(dynamic_cast<const AABBInterface*>(&obj4));
  EXPECT_THAT(candidates, UnorderedElementsAre(&obj3, &obj4));
}

}  // namespace
}  // namespace geometry
}  // namespace nocturne
