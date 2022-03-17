#pragma once

#include <array>
#include <utility>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/line_segment.h"

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

    const Node* Child(int64_t index) const { return children_.at(index); }
    Node* Child(int64_t index) { return children_.at(index); }

    const Node* LChild() const { return children_[0]; }
    Node* LChild() { return children_[0]; }

    const Node* RChild() const { return children_[1]; }
    Node* RChild() { return children_[1]; }

   protected:
    AABB aabb_;
    const AABBInterface* object_ = nullptr;
    std::array<Node*, 2> children_ = {nullptr, nullptr};
  };

  BVH() = default;
  explicit BVH(const std::vector<const AABBInterface*>& objects) {
    InitHierarchy(objects);
  }
  BVH(const std::vector<const AABBInterface*>& objects, int64_t delta)
      : delta_(delta) {
    InitHierarchy(objects);
  }

  bool Empty() const { return nodes_.empty(); }
  int64_t Size() const { return nodes_.size(); }

  void Clear() {
    root_ = nullptr;
    nodes_.clear();
  }

  void InitHierarchy(const std::vector<const AABBInterface*>& objects);

  std::vector<const AABBInterface*> IntersectionCandidates(
      const AABBInterface& object) const {
    std::vector<const AABBInterface*> candidates;
    IntersectionCandidatesImpl(object.GetAABB(), root_, candidates);
    return candidates;
  }

  std::vector<const AABBInterface*> IntersectionCandidates(
      const LineSegment& segment) const {
    std::vector<const AABBInterface*> candidates;
    IntersectionCandidatesImpl(segment, root_, candidates);
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

  void IntersectionCandidatesImpl(
      const AABB& aabb, const Node* cur,
      std::vector<const AABBInterface*>& candidates) const;

  void IntersectionCandidatesImpl(
      const LineSegment& segment, const Node* cur,
      std::vector<const AABBInterface*>& candidates) const;

  std::vector<Node> nodes_;
  Node* root_ = nullptr;
  const int64_t delta_ = 4;
};

}  // namespace geometry
}  // namespace nocturne
