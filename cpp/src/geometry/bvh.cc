#include "geometry/bvh.h"

#include <algorithm>

#include "geometry/morton.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

namespace {

// Binary search for smallest index who shares the same highest bit with r - 1
// in range [l, r).
int64_t FindPivot(
    const std::vector<std::tuple<uint64_t, const AABBInterface*>>& objects,
    int64_t l, int64_t r) {
  const uint64_t last = std::get<0>(objects[r - 1]);
  const int64_t pivot_prefix = __builtin_clzll(std::get<0>(objects[l]) ^ last);
  int64_t ret = r;
  while (l < r) {
    const int64_t mid = l + (r - l) / 2;
    const int64_t cur_prefix =
        __builtin_clzll(std::get<0>(objects[mid]) ^ last);
    if (cur_prefix > pivot_prefix) {
      ret = mid;
      r = mid;
    } else {
      l = mid + 1;
    }
  }
  return ret;
}

}  // namespace

void BVH::InitHierarchy(const std::vector<const AABBInterface*>& objects) {
  Clear();
  const int64_t n = objects.size();
  nodes_.reserve(2 * n - 1);

  std::vector<std::tuple<uint64_t, const AABBInterface*>> encoded_objects;
  encoded_objects.reserve(n);
  for (const auto* obj : objects) {
    const AABB aabb = obj->GetAABB();
    const uint64_t morton_code = morton::Morton2D(aabb.Center());
    encoded_objects.emplace_back(morton_code, obj);
  }
  std::sort(encoded_objects.begin(), encoded_objects.end());
  root_ = InitHierarchyImpl(encoded_objects, /*l=*/0, /*r=*/n);
}

BVH::Node* BVH::InitHierarchyImpl(
    const std::vector<std::tuple<uint64_t, const AABBInterface*>>& objects,
    int64_t l, int64_t r) {
  if (l + 1 == r) {
    return MakeNode(std::get<1>(objects[l]));
  }
  const int64_t p = FindPivot(objects, l, r);
  Node* l_child = InitHierarchyImpl(objects, l, p);
  Node* r_child = InitHierarchyImpl(objects, p, r);
  const AABB aabb = l_child->aabb() || r_child->aabb();
  return MakeNode(aabb, l_child, r_child);
}

void BVH::CollisionCandidatesImpl(
    const AABB& aabb, const Node* cur,
    std::vector<const AABBInterface*>& candidates) const {
  if (!aabb.Overlaps(cur->aabb())) {
    return;
  }
  if (cur->IsLeaf()) {
    candidates.push_back(cur->object());
    return;
  }
  CollisionCandidatesImpl(aabb, cur->LChild(), candidates);
  CollisionCandidatesImpl(aabb, cur->RChild(), candidates);
}

}  // namespace geometry
}  // namespace nocturne
