#include <Simulation.hpp>

#include <SFML/Graphics.hpp>


Simulation::Simulation(bool render) : 
    scenario("../../../scenarios/basic.json"),
    render(render)
{
    if (render) {
        float winWidth = 1500;
        float winHeight = 800;

        sf::ContextSettings settings;
        settings.antialiasingLevel = 8;
        sf::RenderWindow window(sf::VideoMode(winWidth, winHeight), "Nocturne", sf::Style::Default, settings);

        window.setFramerateLimit(60);

        sf::Clock clock;

        std::string fontName = "Arial.ttf";
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
            throw std::invalid_argument("Couldn't load a font file.");
        }

        while (window.isOpen())
        {
            sf::Time elapsed = clock.restart();
            scenario.step(elapsed.asSeconds());
            float fps = 1.0f / elapsed.asSeconds();

            sf::Event event;
            while (window.pollEvent(event))
            {
                if (event.type == sf::Event::Closed) {
                    window.close();
                } else if (event.type == sf::Event::Resized) {
                    window.setView(getView(sf::Vector2u(event.size.width, event.size.height)));
                }
           }

            window.clear(sf::Color(50, 50, 50));

            window.setView(getView(window.getSize()));

            sf::Transform transform;
            transform.scale(1, -1); // horizontal flip
            window.draw(scenario, transform);

            window.setView(sf::View(sf::FloatRect(0, 0, window.getSize().x, window.getSize().y)));

            sf::RectangleShape guiBackground(sf::Vector2f(window.getSize().x, 35));
            guiBackground.setPosition(0, 0);
            guiBackground.setFillColor(sf::Color(0, 0, 0, 100));
            window.draw(guiBackground);

            sf::Text text(std::to_string((int)fps) + " fps", font, 20);
            text.setPosition(10, 5);
            text.setFillColor(sf::Color::White);
            window.draw(text);

            window.display();
        }
    }
}

sf::View Simulation::getView(sf::Vector2u winSize) const {
    sf::FloatRect scenarioBounds = scenario.getBoundingBox();
    scenarioBounds.top = - scenarioBounds.top - scenarioBounds.height;
    
    float padding = 100;
    scenarioBounds.top -= padding;
    scenarioBounds.left -= padding;
    scenarioBounds.width += 2 * padding;
    scenarioBounds.height += 2 * padding;

    sf::Vector2f center = sf::Vector2f(scenarioBounds.left + scenarioBounds.width / 2.0f, scenarioBounds.top + scenarioBounds.height / 2.0f);
    sf::Vector2f size = sf::Vector2f(winSize.x, winSize.y) *
        std::max(scenarioBounds.width / winSize.x, scenarioBounds.height / winSize.y);
    sf::View view(center, size);
    
    return view;
}

void Simulation::reset() {
    std::cout << "Resetting simulation." << std::endl;
}

void Simulation::getCircle() const {
    sf::RenderTexture texture;

    texture.create(4, 5);
    texture.clear(sf::Color::Red);
    texture.display();

    sf::Texture text = texture.getTexture();

    sf::Image img = text.copyToImage();

    // img.saveToFile("test.png");

    const unsigned char* pixels = img.getPixelsPtr();

    // for (int i = 0; i < 80; i++)
    //     std::cout << (int)pixels[i] << std::endl;

    // cf https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html convert to a numpy array
}

/*
sf::Texture texture;
texture.create(render_window.getSize().x, render_window.getSize().y);
texture.update(render_window);
if (texture.copyToImage().saveToFile(filename))
{
    std::cout << "screenshot saved to " << filename << std::endl;
}
*/