#pragma once

#include <SFML/Graphics.hpp>

#include "geometry/vector_2d.h"

namespace nocturne {
namespace utils {

// Loads a font file `font_name` from the system (eg Arial.ttf).
// Font files are currently searched in standard Linux, macOS and Windows paths.
sf::Font LoadFont(const std::string& font_name);

inline sf::Vector2f ToVector2f(const geometry::Vector2D& vec,
                               bool flip_y = false) {
  return sf::Vector2f(vec.x(), flip_y ? -vec.y() : vec.y());
}

}  // namespace utils
}  // namespace nocturne
