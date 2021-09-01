#pragma once

#include <vector>

class Road;
class Vehicle;


class Network {
public:
    Network();

    Lane* getVehicleLane(Vehicle veh);

private:
    std::vector<Road> roads;
};