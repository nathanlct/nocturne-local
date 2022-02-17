#pragma once

#include <utility>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"

namespace nocturne {
namespace geometry {

// Bounding Volume Hierarchy
// https://en.wikipedia.org/wiki/Bounding_volume_hierarchy
class BVH {
 public:
  class Node {
   public:
    Node() = default;
    explicit Node(const AABBInterface* object)
        : aabb_(object->GetAABB()), object_(object) {}
    Node(const AABB& aabb, const AABBInterface* object, Node* l_child,
         Node* r_child)
        : aabb_(aabb), object_(object) {
      children_[0] = l_child;
      children_[1] = r_child;
    }

    const AABB& aabb() const { return aabb_; }
    const AABBInterface* object() const { return object_; }

    bool IsLeaf() const { return object_ != nullptr; }

    const Node* LChild() const { return children_[0]; }
    Node* LChild() { return children_[0]; }

    const Node* RChild() const { return children_[1]; }
    Node* RChild() { return children_[1]; }

   protected:
    AABB aabb_;
    const AABBInterface* object_ = nullptr;
    Node* children_[2] = {nullptr, nullptr};
  };

  BVH() = default;
  BVH(const std::vector<const AABBInterface*>& objects) {
    InitHierarchy(objects);
  }

  void Clear() {
    root_ = nullptr;
    nodes_.clear();
  }

  void InitHierarchy(const std::vector<const AABBInterface*>& objects);

  std::vector<const AABBInterface*> CollisionCandidates(
      const AABBInterface* object) const {
    std::vector<const AABBInterface*> candidates;
    CollisionCandidatesImpl(object->GetAABB(), root_, candidates);
    return candidates;
  }

 protected:
  Node* MakeNode(const AABBInterface* object) {
    nodes_.emplace_back(object);
    return &nodes_.back();
  }

  Node* MakeNode(Node* l_child, Node* r_child) {
    nodes_.emplace_back((l_child->aabb() || r_child->aabb()),
                        /*object=*/nullptr, l_child, r_child);
    return &nodes_.back();
  }

  std::vector<BVH::Node*> CombineNodes(const std::vector<BVH::Node*>& nodes,
                                       int64_t num);

  // Init hierarchy in range [l, r).
  std::vector<Node*> InitHierarchyImpl(
      const std::vector<std::pair<uint64_t, const AABBInterface*>>& objects,
      int64_t l, int64_t r);

  void CollisionCandidatesImpl(
      const AABB& aabb, const Node* cur,
      std::vector<const AABBInterface*>& candidates) const;

  std::vector<Node> nodes_;
  Node* root_ = nullptr;
};

}  // namespace geometry
}  // namespace nocturne
