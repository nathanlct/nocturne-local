#include "../cpp/include/nocturne_bits/Simulation.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_simulation(py::module &m) {
    m.doc() = "nocturne documentation for class Simulation";

    py::class_<Simulation>(m, "Simulation")
        .def(
            py::init<bool, std::string>(), 
            "Constructor for Scenario",
            py::arg("render") = false, 
            py::arg("scenarioPath") = "");
}