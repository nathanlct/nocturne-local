#include <pybind11/pybind11.h>
// #include <pybind11/stl.h>  // automatic conversion from C++ std::vector to
// Python list

#include "Simulation.hpp"

namespace py = pybind11;

void init_simulation(py::module& m) {
  m.doc() = "nocturne documentation for class Simulation";

  py::class_<nocturne::Simulation>(m, "Simulation")
      .def(py::init<std::string, int, bool>(), "Constructor for Simulation",
           py::arg("scenarioPath") = "",
           py::arg("startTime") = 0,
           py::arg("useNonVehicles") = true)
      .def("step", &nocturne::Simulation::step)
      .def("render", &nocturne::Simulation::render)
      .def("reset", &nocturne::Simulation::reset)
      .def("saveScreenshot", &nocturne::Simulation::saveScreenshot)
      .def("getScenario", &nocturne::Simulation::getScenario);
}
