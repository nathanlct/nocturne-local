#include <Simulation.hpp>
#include <iostream>

#include <SFML/Graphics.hpp>


Simulation::Simulation(bool render) : scenario(), render(render) {
    if (render) {
        sf::ContextSettings settings;
        settings.antialiasingLevel = 8;
        sf::RenderWindow window(sf::VideoMode(800, 800), "Nocturne", sf::Style::Default, settings);
        window.setFramerateLimit(30);

        while (window.isOpen())
        {
            sf::Event event;
            while (window.pollEvent(event))
            {
                if (event.type == sf::Event::Closed)
                    window.close();
            }

            window.clear(sf::Color(50, 50, 50));

            sf::Transform flipHorizontally;
            flipHorizontally.scale(1, -1);
            flipHorizontally.translate(0, -800);
            window.draw(scenario, flipHorizontally);

            window.display();
        }
    }
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