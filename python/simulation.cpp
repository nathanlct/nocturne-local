#include <pybind11/pybind11.h>
// #include <pybind11/stl.h>  // automatic conversion from C++ std::vector to
// Python list

#include "Simulation.hpp"

namespace py = pybind11;

void init_simulation(py::module& m) {
  m.doc() = "nocturne documentation for class Simulation";

  py::class_<nocturne::Simulation>(m, "Simulation")
      .def(py::init<std::string, int, bool>(), "Constructor for Simulation",
           py::arg("scenario_path") = "", py::arg("start_time") = 0,
           py::arg("use_non_vehicles") = true)
      .def("step", &nocturne::Simulation::step)
      .def("render", &nocturne::Simulation::render)
      .def("reset", &nocturne::Simulation::reset)
      .def("saveScreenshot", &nocturne::Simulation::saveScreenshot)
      .def("getScenario", &nocturne::Simulation::getScenario,
           py::return_value_policy::reference);
}
