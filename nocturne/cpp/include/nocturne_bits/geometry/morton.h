#pragma once

#include <cstdint>

#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {
namespace morton {

uint64_t Morton2D(const Vector2D& v);

}  // namespace morton
}  // namespace geometry
}  // namespace nocturne
