#include "utils/sf_utils.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace nocturne {
namespace utils {

namespace {

#if defined(_APPLE_) && defined(_MACH_)

std::vector<std::string> GetFontPaths() {
  return {"/System/Library/Fonts/Supplemental/", "~/Library/Fonts/"};
}

#else  // defined(_APPLE_) && defined(_MACH_)

std::vector<std::string> GetFontPaths() {
  const std::string username = std::getenv("USER");
  return {"/usr/share/fonts", "/usr/local/share/fonts",
          "/home/" + username + "/.fonts/",
          "/private/home/" + username + "/.fonts/"};
}

#endif  // defined(_APPLE_) && defined(_MACH_)

bool FindFontPath(const std::string& font_name, std::string& font_path) {
  const std::vector<std::string> font_paths = GetFontPaths();
  for (const std::string& fp : font_paths) {
    const std::string cur_path = fp + font_name;
    std::ifstream font(cur_path);
    if (font.is_open()) {
      font_path = cur_path;
      return true;
    }
  }
  return false;
}

}  // namespace

sf::Font GetFont(const std::string& font_name) {
  std::string font_path;
  sf::Font font;
  if (!FindFontPath(font_name, font_path) || !font.loadFromFile(font_path)) {
    throw std::invalid_argument("Could not load font file " + font_name + ".");
  }
  return font;
}

}  // namespace utils
}  // namespace nocturne
