#include "simulation.h"

#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace nocturne {

void DefineSimulation(py::module& m) {
  py::class_<Simulation, std::shared_ptr<Simulation>>(m, "Simulation")
      .def(py::init<const std::string&,
                    const std::unordered_map<std::string, std::string>&>(),
           "Constructor for Simulation", py::arg("scenario_path") = "",
           py::arg("config"))
      .def(py::init([](const std::string& scenario_path, py::dict config) {
             std::unordered_map<std::string, std::string> config_map;
             for (auto item : config) {
               config_map[std::string(py::str(item.first))] =
                   std::string(py::str(item.second));
             }
             return std::unique_ptr<Simulation>(
                 new Simulation(scenario_path, config_map));
           }),
           py::arg("scenario_path") = "", py::arg("config") = py::dict())
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
