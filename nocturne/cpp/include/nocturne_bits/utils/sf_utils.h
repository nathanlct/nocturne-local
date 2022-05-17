#pragma once

#include <SFML/Graphics.hpp>

#include "geometry/vector_2d.h"

namespace nocturne {
namespace utils {

// @brief Load a font file from the system.
// Font files are currently searched in standard Linux and macOS paths.
//
// @param fontName name of the font file (eg "Arial.ttf")
// @return sf::Font
// @throw std::invalid_argument if font file is not found
sf::Font GetFont(const std::string& font_name);

inline sf::Vector2f ToVector2f(const geometry::Vector2D& vec,
                               bool flip_y = false) {
  return sf::Vector2f(vec.x(), flip_y ? -vec.y() : vec.y());
}

}  // namespace utils
}  // namespace nocturne
