#pragma once

#include <SFML/Graphics.hpp>

#include "ndarray.h"

namespace nocturne {

class Canvas : public sf::RenderTexture {
 public:
  Canvas(int64_t width, int64_t height,
         sf::Color background_color = sf::Color(50, 50, 50))
      : width_(width), height_(height) {
    sf::ContextSettings texture_settings;
    texture_settings.antialiasingLevel = 4;
    create(width, height, texture_settings);
    clear(background_color);
  }

  NdArray<unsigned char> AsNdArray() {
    display();
    const sf::Image image = getTexture().copyToImage();
    const unsigned char* pixels = (const unsigned char*)image.getPixelsPtr();
    return NdArray<unsigned char>({height_, width_, /*channels=*/int64_t(4)},
                                  pixels);
  }

 private:
  const int64_t width_;
  const int64_t height_;
};

}  // namespace nocturne
