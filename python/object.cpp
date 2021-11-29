#include "../cpp/include/nocturne_bits/Object.hpp"
#include "../cpp/include/nocturne_bits/Vehicle.hpp"
#include <pybind11/pybind11.h>
// #include <pybind11/stl.h>  // automatic conversion from C++ std::vector to Python list

namespace py = pybind11;

void init_object(py::module &m) {
    m.doc() = "nocturne documentation for class Object and subclass Vehicle";

    py::class_<Object, std::shared_ptr<Object>>(m, "Object")
        .def("getWidth", &Object::getWidth)
        .def("getLength", &Object::getLength)
        .def("getPosition", &Object::getPosition)
        .def("getGoalPosition", &Object::getGoalPosition)
        .def("getSpeed", &Object::getSpeed)
        .def("getHeading", &Object::getHeading)
        .def("getID", &Object::getID)
        .def("getType", &Object::getType)
        .def("getCollided", &Object::getCollided)
        .def("setPosition", &Object::setPosition)
        .def("setGoalPosition", &Object::setGoalPosition)
        .def("setSpeed", &Object::setSpeed)
        .def("setHeading", &Object::setHeading);

    py::class_<Vehicle, std::shared_ptr<Vehicle>, Object>(m, "Vehicle")
        .def("getWidth", &Vehicle::getWidth)
        .def("getPosition", &Vehicle::getPosition)
        .def("getGoalPosition", &Vehicle::getGoalPosition)
        .def("getSpeed", &Vehicle::getSpeed)
        .def("getHeading", &Vehicle::getHeading)
        .def("getID", &Vehicle::getID)
        .def("getType", &Vehicle::getType)
        .def("getCollided", &Vehicle::getCollided)
        .def("setAccel", &Vehicle::setAccel)
        .def("setSteeringAngle", &Vehicle::setSteeringAngle)
        .def("setPosition", &Vehicle::setPosition)
        .def("setGoalPosition", &Vehicle::setGoalPosition)
        .def("setSpeed", &Vehicle::setSpeed)
        .def("setHeading", &Vehicle::setHeading);
}
