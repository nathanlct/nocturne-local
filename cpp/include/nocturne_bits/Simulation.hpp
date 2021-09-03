#pragma once

#include "Scenario.hpp"


class Simulation {
public:
    Simulation(bool render);

    void reset();
    void step();

    void getCircle() const; // tmp

private:
    Scenario scenario;

    bool render;
};