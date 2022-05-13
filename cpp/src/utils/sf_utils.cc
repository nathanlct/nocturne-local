#include "utils/sf_utils.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace nocturne {
namespace utils {

namespace {

#if defined(__APPLE__)  // OSX

std::vector<std::string> GetFontPaths() {
  return {"/System/Library/Fonts/Supplemental/", "~/Library/Fonts/"};
}

#elif defined(_WIN32)  // Windows 32 bit or 64 bit

std::vector<std::string> GetFontPaths() {
  return {"C:/Windows/Fonts/"};
}

#else  // Linux

std::vector<std::string> GetFontPaths() {
  const std::string username = std::getenv("USER");
  return {"/usr/share/fonts", "/usr/local/share/fonts",
          "/home/" + username + "/.fonts/",
          "/private/home/" + username + "/.fonts/"};
}

#endif

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

sf::Font LoadFont(const std::string& font_name) {
  std::string font_path;
  sf::Font font;
  if (!FindFontPath(font_name, font_path) || !font.loadFromFile(font_path)) {
    throw std::invalid_argument("Could not load font file " + font_name + ".");
  }
  return font;
}

}  // namespace utils
}  // namespace nocturne
