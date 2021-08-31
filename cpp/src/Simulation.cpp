#include "../include/nocturne_bits/Simulation.hpp"
#include <iostream>
#include <SFML/Graphics.hpp>

Simulation::Simulation() {
        
}

void Simulation::reset() const {
    std::cout << "Resetting simulation." << std::endl;

    sf::RenderWindow window(sf::VideoMode(200, 200), "SFML works!");
    sf::CircleShape shape(100.f);
    shape.setFillColor(sf::Color::Green);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(shape);
        window.display();
    }
}

