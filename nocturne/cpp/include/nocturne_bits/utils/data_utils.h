#pragma once

#include <algorithm>
#include <vector>

namespace nocturne {
namespace utils {

template <typename MaskType, typename DataType>
int64_t MaskedPartition(const std::vector<MaskType>& mask,
                        std::vector<DataType>& data) {
  const int64_t n = data.size();
  int64_t pivot = 0;
  for (; pivot < n && static_cast<bool>(mask[pivot]); ++pivot)
    ;
  for (int64_t i = pivot + 1; i < n; ++i) {
    if (static_cast<bool>(mask[i])) {
      std::swap(data[i], data[pivot++]);
    }
  }
  return pivot;
}

// std::partial_sort use heap_sort which slower than std::sort especially when k
// is large.
template <class RandomIt, class CompareFunc>
void PartialSort(RandomIt first, RandomIt middle, RandomIt last,
                 CompareFunc cmp) {
  std::nth_element(first, middle, last, cmp);
  std::sort(first, middle, cmp);
}

}  // namespace utils
}  // namespace nocturne
