#pragma once


class Scenario;

class Simulation {
public:
    Simulation();

    void reset();
    void step();

    void getCircle() const; // tmp

private:
    Scenario scenario;
};