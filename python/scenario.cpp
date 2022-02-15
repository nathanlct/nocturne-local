#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // automatic conversion from C++ std::vector to Python list

#include "Scenario.hpp"

namespace py = pybind11;

void init_scenario(py::module& m) {
  m.doc() = "nocturne documentation for class Scenario";

  py::class_<nocturne::Scenario>(m, "Scenario")
      .def("getRoadObjects", &nocturne::Scenario::getRoadObjects)
      .def("getVehicles", &nocturne::Scenario::getVehicles)
      .def("getCone", &nocturne::Scenario::getCone)
      .def("getImage", &nocturne::Scenario::getImage,
           "Return a numpy array of dimension (w, h, 4) representing the scene",
           py::arg("object") = nullptr, py::arg("renderGoals") = false)
      .def("removeObject", &nocturne::Scenario::removeObject)
      .def("createVehicle", &nocturne::Scenario::createVehicle)
      .def("isVehicleOnRoad", &nocturne::Scenario::isVehicleOnRoad)
      .def("isPointOnRoad", &nocturne::Scenario::isPointOnRoad);
  // .def(
  //     py::init<std::string>(),
  //     "Constructor for Scenario",
  //     py::arg("scenarioPath") = "")
}
