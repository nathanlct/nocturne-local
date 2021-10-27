#include "../cpp/include/nocturne_bits/Object.hpp"
#include "../cpp/include/nocturne_bits/Vehicle.hpp"
#include <pybind11/pybind11.h>
// #include <pybind11/stl.h>  // automatic conversion from C++ std::vector to Python list

namespace py = pybind11;

void init_object(py::module &m) {
    m.doc() = "nocturne documentation for class Object and subclass Vehicle";

    py::class_<Object>(m, "Object")
        .def("getWidth", &Object::getWidth, "Get object width");

    py::class_<Vehicle, Object>(m, "Vehicle")
        .def("getWidth", &Vehicle::getWidth, "Get vehicle width")
        .def("setAccel", &Vehicle::setAccel)
        .def("setSteeringAngle", &Vehicle::setSteeringAngle);
}
