#include "../cpp/include/nocturne_bits/Simulation.hpp"
#include <pybind11/pybind11.h>
// #include <pybind11/stl.h>  // automatic conversion from C++ std::vector to Python list

namespace py = pybind11;

void init_simulation(py::module &m) {
    m.doc() = "nocturne documentation for class Simulation";

    py::class_<Simulation>(m, "Simulation")
        .def(
            py::init<std::string>(), 
            "Constructor for Scenario",
            py::arg("scenarioPath") = "")
        .def("step", &Simulation::step)
        .def("render", &Simulation::render)
        .def("getScenario", &Simulation::getScenario);
}