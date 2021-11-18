#include "../cpp/include/nocturne_bits/Scenario.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // automatic conversion from C++ std::vector to Python list

namespace py = pybind11;

void init_scenario(py::module &m) {
    m.doc() = "nocturne documentation for class Scenario";

    py::class_<Scenario>(m, "Scenario")
        .def("getRoadObjects", &Scenario::getRoadObjects)
        .def("getVehicles", &Scenario::getVehicles)
        .def("getCone", &Scenario::getCone)
        .def("getImage", &Scenario::getImage, 
             "Return a numpy array of dimension (w, h, 4) representing the scene",
             py::arg("object") = nullptr, py::arg("renderGoals") = false)
        .def("removeObject", &Scenario::removeObject)
        .def("createVehicle", &Scenario::createVehicle)
        .def("isVehicleOnRoad", &Scenario::isVehicleOnRoad);
        // .def(
        //     py::init<std::string>(), 
        //     "Constructor for Scenario",
        //     py::arg("scenarioPath") = "")
}