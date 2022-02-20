#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // automatic conversion from C++ std::vector to Python list

#include "Scenario.hpp"

namespace py = pybind11;

void init_scenario(py::module& m) {
  m.doc() = "nocturne documentation for class Scenario";

  py::class_<nocturne::Scenario>(m, "Scenario")
      .def(py::init<std::string, int, bool>(), "Constructor for Scenario",
           py::arg("path") = "",
           py::arg("startTime") = 0,
           py::arg("useNonVehicles") = true)
      .def("getVehicles", &nocturne::Scenario::getVehicles)
      .def("getMaxEnvTime", &nocturne::Scenario::getMaxEnvTime)
      .def("getRoadLines", &nocturne::Scenario::getRoadLines)
      .def("getCone", &nocturne::Scenario::getCone,
          "Draw a cone representing the objects that the agent can see",
          py::arg("object"), py::arg("viewAngle") = 1.58, py::arg("headTilt") = 0.0, py::arg("obscuredView") = true)
      .def("getImage", &nocturne::Scenario::getImage,
           "Return a numpy array of dimension (w, h, 4) representing the scene",
           py::arg("object") = nullptr, py::arg("renderGoals") = false)
      .def("removeVehicle", &nocturne::Scenario::removeVehicle)
      .def("createVehicle", &nocturne::Scenario::createVehicle)
      .def("hasExpertAction", &nocturne::Scenario::hasExpertAction)
      .def("getExpertAction", &nocturne::Scenario::getExpertAction)
      .def("getValidExpertStates", &nocturne::Scenario::getValidExpertStates);
  // .def(
  //     py::init<std::string>(),
  //     "Constructor for Scenario",
  //     py::arg("scenarioPath") = "")
}
