#include "Scenario.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numpy_utils.h"

namespace py = pybind11;

namespace nocturne {

pybind11::array_t<float> Scenario::egoStateObservation(
    const KineticObject& src) const {
  return utils::AsNumpyArray<float>(egoObservationImpl(src));
}

pybind11::array_t<float> Scenario::Observation(const KineticObject& src,
                                               float view_dist,
                                               float view_angle) const {
  return utils::AsNumpyArray<float>(
      ObservationImpl(src, view_dist, view_angle));
}

}  // namespace nocturne

void init_scenario(py::module& m) {
  m.doc() = "nocturne documentation for class Scenario";

  py::class_<nocturne::Scenario>(m, "Scenario")
      .def(py::init<std::string, int, bool>(), "Constructor for Scenario",
           py::arg("path") = "", py::arg("start_time") = 0,
           py::arg("use_non_vehicles") = true)
      .def("getVehicles", &nocturne::Scenario::getVehicles,
           py::return_value_policy::reference)
      .def("getPedestrians", &nocturne::Scenario::getPedestrians,
           py::return_value_policy::reference)
      .def("getCyclists", &nocturne::Scenario::getCyclists,
           py::return_value_policy::reference)
      .def("getObjectsThatMoved", &nocturne::Scenario::getObjectsThatMoved,
           py::return_value_policy::reference)
      .def("getMaxEnvTime", &nocturne::Scenario::getMaxEnvTime)
      .def("getRoadLines", &nocturne::Scenario::getRoadLines)
      .def("getCone", &nocturne::Scenario::getCone,
           "Draw a cone representing the objects that the agent can see",
           py::arg("object"), py::arg("view_dist") = 60.0,
           py::arg("view_angle") = 1.570796, py::arg("head_tilt") = 0.0,
           py::arg("obscuredView") = true)
      .def("getImage", &nocturne::Scenario::getImage,
           "Return a numpy array of dimension (w, h, 4) representing the scene",
           py::arg("object") = nullptr, py::arg("render_goals") = false)
      .def("removeVehicle", &nocturne::Scenario::removeVehicle)
      // .def("createVehicle", &nocturne::Scenario::createVehicle)
      .def("hasExpertAction", &nocturne::Scenario::hasExpertAction)
      .def("getExpertAction", &nocturne::Scenario::getExpertAction)
      .def("getExpertSpeeds", &nocturne::Scenario::getExpertSpeeds,
           py::arg("timeIndex"), py::arg("vehIndex"))
      .def("getValidExpertStates", &nocturne::Scenario::getValidExpertStates)
      .def("getMaxNumVisibleKineticObjects",
           &nocturne::Scenario::getMaxNumVisibleKineticObjects)
      .def("getMaxNumVisibleRoadPoints",
           &nocturne::Scenario::getMaxNumVisibleRoadPoints)
      .def("getMaxNumVisibleStopSigns",
           &nocturne::Scenario::getMaxNumVisibleStopSigns)
      .def("getMaxNumVisibleTrafficLights",
           &nocturne::Scenario::getMaxNumVisibleTrafficLights)
      .def("getKineticObjectFeatureSize",
           &nocturne::Scenario::getKineticObjectFeatureSize)
      .def("getRoadPointFeatureSize",
           &nocturne::Scenario::getRoadPointFeatureSize)
      .def("getTrafficLightFeatureSize",
           &nocturne::Scenario::getTrafficLightFeatureSize)
      .def("getStopSignsFeatureSize",
           &nocturne::Scenario::getStopSignsFeatureSize)
      .def("getEgoFeatureSize", &nocturne::Scenario::getEgoFeatureSize)
      // .def("getVisibleObjects", &nocturne::Scenario::getVisibleObjects)
      // .def("getVisibleRoadPoints", &nocturne::Scenario::getVisibleRoadPoints)
      // .def("getVisibleStopSigns", &nocturne::Scenario::getVisibleStopSigns)
      // .def("getVisibleTrafficLights",
      // &nocturne::Scenario::getVisibleTrafficLights)
      .def("observation", &nocturne::Scenario::Observation, py::arg("object"),
           py::arg("view_dist") = 60, py ::arg("view_angle") = 1.58)
      .def("egoStateObservation", &nocturne::Scenario::egoStateObservation);
  // .def(
  //     py::init<std::string>(),
  //     "Constructor for Scenario",
  //     py::arg("scenarioPath") = "")
}
