#pragma once

#include <SFML/Graphics.hpp>
#include <fstream>

#include "geometry/vector_2d.h"

namespace nocturne {
namespace utils {

/**
 * @brief Load a font file from the system.
 * Font files are currently searched in standard Linux and macOS paths.
 *
 * @param fontName name of the font file (eg "Arial.ttf")
 * @return sf::Font
 * @throw std::invalid_argument if font file is not found
 */
inline sf::Font getFont(std::string fontName) {
  std::string username = getenv("USER");
  std::vector<std::string> fontPaths = {
      // OS X
      "/System/Library/Fonts/Supplemental/", "~/Library/Fonts/",
      // Linux
      "/usr/share/fonts", "/usr/local/share/fonts", "~/.fonts/",
      "/private/home/" + username + "/.fonts/"};

  std::string fontPath = "";
  for (std::string& fp : fontPaths) {
    std::string path = fp + fontName;
    std::ifstream data(path);
    if (data.is_open()) {
      fontPath = path;
      break;
    }
  }

  sf::Font font;
  if (fontPath == "" || !font.loadFromFile(fontPath)) {
    throw std::invalid_argument("could not load font file " + fontName + ".");
  }
  return font;
}

inline sf::Vector2f ToVector2f(const geometry::Vector2D& vec,
                               bool flip_y = false) {
  return sf::Vector2f(vec.x(), flip_y ? -vec.y() : vec.y());
}

}  // namespace utils
}  // namespace nocturne
