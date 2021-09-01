#pragma once

#include <vector>

class Lane;


class Road {
public:
    Road();

private:
    std::vector<Lane> lanes;

    float length;
};