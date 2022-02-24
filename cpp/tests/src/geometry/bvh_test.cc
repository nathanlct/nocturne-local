#include "geometry/bvh.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/line_segment.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {
namespace {

using testing::ElementsAreArray;
using testing::UnorderedElementsAre;

class MockObject : public AABBInterface {
 public:
  MockObject(int64_t id, const Vector2D& center, float radius)
      : id_(id), center_(center), radius_(radius) {}

  int64_t id() const { return id_; }

  AABB GetAABB() const override {
    return AABB(center_ - radius_, center_ + radius_);
  }

 protected:
  int64_t id_;
  Vector2D center_;
  float radius_;
};

class TestBVH : public BVH {
 public:
  TestBVH(const std::vector<const AABBInterface*>& objects) : BVH(objects) {}
  TestBVH(const std::vector<const AABBInterface*>& objects, int64_t delta)
      : BVH(objects, delta) {}

  std::vector<const AABBInterface*> Leaves() const {
    std::vector<const AABBInterface*> leaves;
    LeavesImpl(root_, leaves);
    return leaves;
  }

  int64_t MaxDepth() const {
    int64_t max_depth = 0;
    MaxDepthImpl(root_, /*cur_depth=*/1, max_depth);
    return max_depth;
  }

 protected:
  void LeavesImpl(const Node* cur,
                  std::vector<const AABBInterface*>& leaves) const {
    if (cur->IsLeaf()) {
      leaves.push_back(cur->object());
      return;
    }
    LeavesImpl(cur->LChild(), leaves);
    LeavesImpl(cur->RChild(), leaves);
  }

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

std::vector<MockObject> MakeRandomObjects(int64_t n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
  std::vector<MockObject> objects;
  objects.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    const float x = dis(gen);
    const float y = dis(gen);
    objects.emplace_back(i, Vector2D(x, y), /*radius=*/1.0f);
  }
  return objects;
}

TEST(BVHTest, InitHierarchyTest) {
  const int64_t n = 100;
  const std::vector<MockObject> objects = MakeRandomObjects(n);
  std::vector<const AABBInterface*> objects_ptr;
  objects_ptr.reserve(n);
  for (const auto& obj : objects) {
    objects_ptr.push_back(dynamic_cast<const AABBInterface*>(&obj));
  }

  TestBVH bvh(objects_ptr);
  const std::vector<const AABBInterface*> leaves = bvh.Leaves();
  std::vector<int64_t> ids;
  ids.reserve(leaves.size());
  for (const AABBInterface* ptr : leaves) {
    ids.push_back(dynamic_cast<const MockObject*>(ptr)->id());
  }
  std::sort(ids.begin(), ids.end());
  std::vector<int64_t> expected_ids(n);
  std::iota(expected_ids.begin(), expected_ids.end(), 0);
  EXPECT_THAT(ids, ElementsAreArray(expected_ids));
}

TEST(BVHTest, InitHierarchyPerfTest) {
  const int64_t n = 1000;
  const std::vector<MockObject> objects = MakeRandomObjects(n);
  std::vector<const AABBInterface*> objects_ptr;
  objects_ptr.reserve(n);
  for (const auto& obj : objects) {
    objects_ptr.push_back(dynamic_cast<const AABBInterface*>(&obj));
  }

  TestBVH bvh(objects_ptr);
  EXPECT_LT(bvh.MaxDepth(), 30);
}

TEST(BVHTest, AABBIntersectionCandidatesTest) {
  const MockObject obj1(1, Vector2D(0.0f, 0.0f), 1.0f);
  const MockObject obj2(2, Vector2D(0.5f, 0.5f), 1.0f);
  const MockObject obj3(3, Vector2D(10.0f, 0.0f), 1.0f);
  const MockObject obj4(4, Vector2D(10.0f, 1.5f), 1.0f);
  const MockObject obj5(5, Vector2D(-10.0f, -10.0f), 1.0f);

  std::vector<const AABBInterface*> objects = {
      dynamic_cast<const AABBInterface*>(&obj1),
      dynamic_cast<const AABBInterface*>(&obj2),
      dynamic_cast<const AABBInterface*>(&obj3),
      dynamic_cast<const AABBInterface*>(&obj4),
      dynamic_cast<const AABBInterface*>(&obj5)};

  TestBVH bvh(objects);
  std::vector<const AABBInterface*> candidates =
      bvh.IntersectionCandidates(obj1);
  EXPECT_THAT(candidates, UnorderedElementsAre(&obj1, &obj2));
  candidates = bvh.IntersectionCandidates(obj2);
  EXPECT_THAT(candidates, UnorderedElementsAre(&obj1, &obj2));
  candidates = bvh.IntersectionCandidates(obj3);
  EXPECT_THAT(candidates, UnorderedElementsAre(&obj3, &obj4));
  candidates = bvh.IntersectionCandidates(obj4);
  EXPECT_THAT(candidates, UnorderedElementsAre(&obj3, &obj4));
}

TEST(BVHTest, LineSegmentIntersectionCandidatesTest) {
  const MockObject obj1(1, Vector2D(0.0f, 0.0f), 1.0f);
  const MockObject obj2(2, Vector2D(0.5f, 0.5f), 1.0f);
  const MockObject obj3(3, Vector2D(10.0f, 0.0f), 1.0f);
  const MockObject obj4(4, Vector2D(10.0f, 1.5f), 1.0f);
  const MockObject obj5(5, Vector2D(-10.0f, -10.0f), 1.0f);

  std::vector<const AABBInterface*> objects = {
      dynamic_cast<const AABBInterface*>(&obj1),
      dynamic_cast<const AABBInterface*>(&obj2),
      dynamic_cast<const AABBInterface*>(&obj3),
      dynamic_cast<const AABBInterface*>(&obj4),
      dynamic_cast<const AABBInterface*>(&obj5)};

  TestBVH bvh(objects);
  std::vector<const AABBInterface*> candidates = bvh.IntersectionCandidates(
      LineSegment(Vector2D(-2.0f, -1.0f), Vector2D(0.0f, -0.5f)));
  EXPECT_THAT(candidates, UnorderedElementsAre(&obj1, &obj2));
  candidates = bvh.IntersectionCandidates(
      LineSegment(Vector2D(9.5f, 0.0f), Vector2D(10.5f, 0.0f)));
  EXPECT_THAT(candidates, UnorderedElementsAre(&obj3));
}

}  // namespace
}  // namespace geometry
}  // namespace nocturne
