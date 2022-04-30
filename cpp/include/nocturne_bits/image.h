#pragma once

#include <cstdint>
#include <vector>

namespace nocturne {

// Utility class for holding a 3D image array that can be converted
// into a numpy array without copy
class Image {
 public:
  Image() = default;
  Image(unsigned char* data, size_t rows, size_t cols, size_t channels = 4)
      : data_(data, data + rows * cols * channels),
        rows_(rows),
        cols_(cols),
        channels_(channels) {}

  const std::vector<unsigned char>& data() const { return data_; }

  const unsigned char* DataPtr() const { return data_.data(); }
  unsigned char* DataPtr() { return data_.data(); }

  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t channels() const { return channels_; }

 protected:
  std::vector<unsigned char> data_;
  size_t rows_;
  size_t cols_;
  size_t channels_;
};

}  // namespace nocturne
