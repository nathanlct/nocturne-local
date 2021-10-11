#include <Simulation.hpp>

#include <SFML/Graphics.hpp>


Simulation::Simulation(bool render) : 
    scenario("../../../scenarios/basic.json"),
    render(render),
    renderTransform()
{
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;

    circleRadius = 400.0f;
    renderedCircleRadius = 200.0f;
    circleTexture.create(2.0f * renderedCircleRadius, 2.0f * renderedCircleRadius, settings);

    renderTransform.scale(1, -1); // horizontal flip

    std::vector<Object*> roadObjects = scenario.getRoadObjects();

    if (render) {
        float winWidth = 1500;
        float winHeight = 800;

        sf::ContextSettings settings;
        settings.antialiasingLevel = 8;
        sf::RenderWindow window(sf::VideoMode(winWidth, winHeight), "Nocturne", sf::Style::Default, settings);

        // window.setFramerateLimit(15);

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

            window.draw(scenario, renderTransform);

            window.setView(sf::View(sf::FloatRect(0, 0, window.getSize().x, window.getSize().y)));

            sf::RectangleShape guiBackgroundTop(sf::Vector2f(window.getSize().x, 35));
            guiBackgroundTop.setPosition(0, 0);
            guiBackgroundTop.setFillColor(sf::Color(0, 0, 0, 100));
            window.draw(guiBackgroundTop);

            sf::Text text(std::to_string((int)fps) + " fps", font, 20);
            text.setPosition(10, 5);
            text.setFillColor(sf::Color::White);
            window.draw(text);

            sf::RectangleShape guiBackgroundRight(sf::Vector2f(renderedCircleRadius + 2.0f * renderedCircleRadius + 40, window.getSize().y));
            guiBackgroundRight.setPosition(window.getSize().x - 2.0f * renderedCircleRadius - 40, 0);
            guiBackgroundRight.setFillColor(sf::Color(0, 0, 0));
            window.draw(guiBackgroundRight);

            renderCone(roadObjects[0], 80.0f * pi / 180.0f, 0.0f * pi / 180.0f);
            sf::Sprite sprite(circleTexture.getTexture());
            sprite.setPosition(window.getSize().x - 2.0f * renderedCircleRadius - 20, 20);
            window.draw(sprite);

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

void Simulation::renderCone(Vector2D center, float heading, float viewAngle, const Object* self) {
    circleTexture.clear(sf::Color(50, 50, 50));
    circleTexture.setView(sf::View(center.toVector2f(true), sf::Vector2f(2.0f * circleRadius, 2.0f * circleRadius)));

    circleTexture.draw(scenario, renderTransform); // todo optimize with objects in range only (quadtree?)

    circleTexture.setView(sf::View(sf::Vector2f(0.0f, 0.0f), sf::Vector2f(circleTexture.getSize())));

    // draw circle
    float r = renderedCircleRadius;
    float diag = std::sqrt(2 * r * r);

    for (int quadrant = 0; quadrant < 4; ++quadrant) {
        std::vector<sf::Vertex> outerCircle; // todo precompute just once

        float angleShift = quadrant * pi / 2.0f;

        Vector2D corner = Vector2D::fromPolar(diag, pi / 4.0f + angleShift);
        outerCircle.push_back(sf::Vertex(corner.toVector2f(), sf::Color::Black));

        int nPoints = 20;
        for (int i = 0; i < nPoints; ++i) {
            float angle = angleShift + i * (pi / 2.0f) / (nPoints - 1);

            Vector2D pt = Vector2D::fromPolar(r, angle);
            outerCircle.push_back(sf::Vertex(pt.toVector2f(), sf::Color::Black));
        }

        circleTexture.draw(&outerCircle[0], outerCircle.size(), sf::TriangleFan, renderTransform);
    }

    // draw cone
    if (viewAngle < 2.0f * pi) {
        std::vector<sf::Vertex> innerCircle; // todo precompute just once

        innerCircle.push_back(sf::Vertex(sf::Vector2f(0.0f, 0.0f), sf::Color::Black));

        float startAngle = heading + viewAngle / 2.0f;
        float endAngle = heading + 2.0f * pi - viewAngle / 2.0f;

        int nPoints = 80; // todo function of angle
        for (int i = 0; i < nPoints; ++i) {
            float angle = startAngle + i * (endAngle - startAngle) / (nPoints - 1);
            Vector2D pt = Vector2D::fromPolar(r, angle);
            innerCircle.push_back(sf::Vertex(pt.toVector2f(), sf::Color::Black));
        }

        circleTexture.draw(&innerCircle[0], innerCircle.size(), sf::TriangleFan, renderTransform);
    }

    // draw obstructions
    std::vector<Object*> roadObjects = scenario.getRoadObjects(); // todo optimize with objects in range only (quadtree?)
    
    for (const Object* obj : roadObjects) {
        if (obj != self) {
            auto lines = obj->getLines();
            for (const auto& [pt1, pt2] : lines) {
                // std::cout << "line " << pt1 << " -> " << pt2 << std::endl;

                int nIntersections = 0;
                for (const auto& [pt3, pt4] : lines) {
                    if (pt1 != pt3 && pt1 != pt4 && Vector2D::doIntersect(pt1, center, pt3, pt4)) {
                        nIntersections++;
                        break;
                        // std::cout << pt1 << " hidden by " << pt3 << " -> " << pt4 << std::endl;
                    }
                }
                for (const auto& [pt3, pt4] : lines) {
                    if (pt2 != pt3 && pt2 != pt4 && Vector2D::doIntersect(pt2, center, pt3, pt4)) {
                        nIntersections++;
                        break;
                        // std::cout << pt1 << " hidden by " << pt3 << " -> " << pt4 << std::endl;
                    }
                }

                if (nIntersections >= 1) {
                    // std::cout << "hide line" << std::endl;

                    sf::ConvexShape hiddenArea;


                    float angle1 = (pt1 - center).angle();
                    float angle2 = (pt2 - center).angle();
                    while (angle2 > angle1) angle2 -= 2.0f * pi;

                    // std::cout << angle1 << " " << angle2 << std::endl;

                    int nPoints = 80; // todo function of angle
                    hiddenArea.setPointCount(nPoints + 2);

                    hiddenArea.setPoint(0, ((pt1 - center) * 0.5f).toVector2f());
                    for (int i = 0; i < nPoints; ++i) {
                        float angle = angle1 + i * (angle2 - angle1) / (nPoints - 1);
                        Vector2D pt = Vector2D::fromPolar(r, angle);
                        hiddenArea.setPoint(1 + i, pt.toVector2f());
                    }
                    hiddenArea.setPoint(nPoints + 1, ((pt2 - center) * 0.5f).toVector2f());

                    hiddenArea.setFillColor(sf::Color::Black);
                    // hiddenArea.setPosition(0, 0); //(pt1 - center).toVector2f());

                    circleTexture.draw(hiddenArea, renderTransform);

                }

                // float angle1 = (pt1 - center).angle();
                // float angle2 = (pt2 - center).angle();

                // if (angle1 < heading && heading < angle2) std::cout << "hidden ";
                // else std::cout << "visible ";
                // std::cout << angle1 << " " << heading << " " << angle2 << std::endl;
                // doIntersect(Vector2D p1, Vector2D q1, Vector2D p2, Vector2D q2)
            }
        }
    }
    
    // Vector2D getPosition() const;
    // float getHeading() const;
    // float getWidth() const;
    // float getHeight() const;
    // std::vector<Vector2D> getCorners() const;



    circleTexture.display();
}   

void Simulation::renderCone(const Object* object, float viewAngle, float headTilt) {
    renderCone(object->getPosition(), object->getHeading() + headTilt, viewAngle, object);
}

// sf::Texture text = texture.getTexture();

// sf::Image img = text.copyToImage();

// // img.saveToFile("test.png");

// const unsigned char* pixels = img.getPixelsPtr();

// for (int i = 0; i < 80; i++)
//     std::cout << (int)pixels[i] << std::endl;

// cf https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html convert to a numpy array
/*
sf::Texture texture;
texture.create(render_window.getSize().x, render_window.getSize().y);
texture.update(render_window);
if (texture.copyToImage().saveToFile(filename))
{
    std::cout << "screenshot saved to " << filename << std::endl;
}
*/

/* 
to draw a circle

Rendering masks are still to be implemented (see issue #1 on github),
 but there's a workaround. Draw your background on a sf::RenderTexture,
 then draw the mask with a transparent color (anything with alpha = 0) 
 on top of it with the sf::BlendNone blending mode. The result is a 
 masked image that you can then draw on your regular scene.

 https://en.sfml-dev.org/forums/index.php?topic=7427.0

sf::BlendMode
sf::Shader

google "sfml render mask"

*/