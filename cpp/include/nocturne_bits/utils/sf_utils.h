#pragma once

#include <SFML/Graphics.hpp>

#include "geometry/vector_2d.h"

namespace nocturne {
namespace utils {

// Loads a font file `font_name` from the system (eg Arial.ttf).
// Font files are currently searched in standard Linux, macOS and Windows paths.
sf::Font LoadFont(const std::string& font_name);

// Creates and returns a pointer to an `sf::CircleShape` object. The circle is
// centered at `position`, has radius `radius` and color `color`.
std::unique_ptr<sf::CircleShape> MakeCircleShape(geometry::Vector2D position,
                                                 float radius, sf::Color color);

// Converts a `geometry::Vector2D` to a `sf::Vector2f`. If `flip_y` is true,
// then the y coordinate is flipped (y becomes -y).
inline sf::Vector2f ToVector2f(const geometry::Vector2D& vec,
                               bool flip_y = false) {
  return sf::Vector2f(vec.x(), flip_y ? -vec.y() : vec.y());
}

}  // namespace utils
}  // namespace nocturne
