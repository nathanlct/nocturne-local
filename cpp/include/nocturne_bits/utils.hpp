#pragma once

/**
 * @brief Load a font file from the system.
 * Font files are currently searched in standard Linux and macOS paths.
 * 
 * @param fontName name of the font file (eg "Arial.ttf")
 * @return sf::Font
 * @throw std::invalid_argument if font file is not found
 */
inline sf::Font getFont(std::string fontName) {
    std::vector<std::string> fontPaths = {
        // OS X
        "/System/Library/Fonts/Supplemental/",
        "~/Library/Fonts/",
        // Linux
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "~/.fonts/"
    };

    std::string fontPath = "";
    for (std::string& fp: fontPaths) {
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
