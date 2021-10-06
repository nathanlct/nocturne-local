#include "../cpp/include/nocturne_bits/Simulation.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_simulation(py::module &m) {
    py::class_<Simulation>(m, "Simulation")
    .def(py::init<bool>());
}