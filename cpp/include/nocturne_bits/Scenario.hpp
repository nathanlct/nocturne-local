#pragma once

#include <string>

class Network;


class Scenario {
public:
    Scenario();

    void load_from_xml(std::string path);
    void create();  // needs to insert vehicles etc from XML config

private:
    Network roadNetwork;
};