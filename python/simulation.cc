#include "Simulation.hpp"

#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;

namespace nocturne {

void DefineSimulation(py::module& m) {
  m.doc() = "nocturne documentation for class Simulation";

  py::class_<Simulation, std::shared_ptr<Simulation>>(m, "Simulation")
      .def(py::init<std::string, int, bool>(), "Constructor for Simulation",
           py::arg("scenario_path") = "", py::arg("start_time") = 0,
           py::arg("use_non_vehicles") = true)
      .def("step", &Simulation::step)
      .def("render", &Simulation::render)
      .def("reset", &Simulation::reset)
      .def("saveScreenshot", &Simulation::saveScreenshot)
      .def("getScenario", &Simulation::getScenario,
           py::return_value_policy::reference);
}

}  // namespace nocturne
