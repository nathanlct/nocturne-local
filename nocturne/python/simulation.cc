#include "simulation.h"

#include <pybind11/pybind11.h>

#include <memory>
#include <string>

namespace py = pybind11;

namespace nocturne {

void DefineSimulation(py::module& m) {
  py::class_<Simulation, std::shared_ptr<Simulation>>(m, "Simulation")
      .def(py::init<const std::string&, int, bool, bool>(),
           "Constructor for Simulation", py::arg("scenario_path") = "",
           py::arg("start_time") = 0, py::arg("allow_non_vehicles") = true,
           py::arg("spawn_invalid_objects") = false)
      .def("reset", &Simulation::Reset)
      .def("step", &Simulation::Step)
      .def("render", &Simulation::Render)
      .def("scenario", &Simulation::GetScenario,
           py::return_value_policy::reference)
      .def("save_screenshot", &Simulation::SaveScreenshot)

      // TODO: Deprecate the legacy methods below.
      .def("saveScreenshot", &Simulation::SaveScreenshot)
      .def("getScenario", &Simulation::GetScenario,
           py::return_value_policy::reference);
}

}  // namespace nocturne
