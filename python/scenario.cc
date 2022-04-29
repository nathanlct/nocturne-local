#include "Scenario.hpp"

#include <pybind11/pybind11.h>

#include <memory>

#include "geometry/geometry_utils.h"
#include "numpy_utils.h"

namespace py = pybind11;

namespace nocturne {

using geometry::utils::kHalfPi;

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

void DefineScenario(py::module& m) {
  m.doc() = "nocturne documentation for class Scenario";

  py::class_<Scenario, std::shared_ptr<Scenario>>(m, "Scenario")
      .def(py::init<std::string, int, bool>(), "Constructor for Scenario",
           py::arg("path") = "", py::arg("start_time") = 0,
           py::arg("use_non_vehicles") = true)
      .def("getVehicles", &Scenario::getVehicles,
           py::return_value_policy::reference)
      .def("getPedestrians", &Scenario::getPedestrians,
           py::return_value_policy::reference)
      .def("getCyclists", &Scenario::getCyclists,
           py::return_value_policy::reference)
      .def("getObjectsThatMoved", &Scenario::getObjectsThatMoved,
           py::return_value_policy::reference)
      .def("getMaxEnvTime", &Scenario::getMaxEnvTime)
      .def("getRoadLines", &Scenario::getRoadLines)
      .def("getCone", &Scenario::getCone,
           "Draw a cone representing the objects that the agent can see",
           py::arg("object"), py::arg("view_dist") = 60.0,
           py::arg("view_angle") = kHalfPi, py::arg("head_tilt") = 0.0,
           py::arg("obscuredView") = true)
      .def("getImage", &Scenario::getImage,
           "Return a numpy array of dimension (w, h, 4) representing the scene",
           py::arg("object") = nullptr, py::arg("render_goals") = false)
      .def("removeVehicle", &Scenario::removeVehicle)
      .def("hasExpertAction", &Scenario::hasExpertAction)
      .def("getExpertAction", &Scenario::getExpertAction)
      .def("getValidExpertStates", &Scenario::getValidExpertStates)
      .def("getMaxNumVisibleKineticObjects",
           &Scenario::getMaxNumVisibleKineticObjects)
      .def("getMaxNumVisibleRoadPoints", &Scenario::getMaxNumVisibleRoadPoints)
      .def("getMaxNumVisibleStopSigns", &Scenario::getMaxNumVisibleStopSigns)
      .def("getMaxNumVisibleTrafficLights",
           &Scenario::getMaxNumVisibleTrafficLights)
      .def("getKineticObjectFeatureSize",
           &Scenario::getKineticObjectFeatureSize)
      .def("getRoadPointFeatureSize", &Scenario::getRoadPointFeatureSize)
      .def("getTrafficLightFeatureSize", &Scenario::getTrafficLightFeatureSize)
      .def("getStopSignsFeatureSize", &Scenario::getStopSignsFeatureSize)
      .def("getEgoFeatureSize", &Scenario::getEgoFeatureSize)
      // .def("getVisibleObjects", &Scenario::getVisibleObjects)
      // .def("getVisibleRoadPoints", &Scenario::getVisibleRoadPoints)
      // .def("getVisibleStopSigns", &Scenario::getVisibleStopSigns)
      // .def("getVisibleTrafficLights", &Scenario::getVisibleTrafficLights)
      .def("observation", &Scenario::Observation, py::arg("object"),
           py::arg("view_dist") = 60, py ::arg("view_angle") = kHalfPi)
      .def("egoStateObservation", &Scenario::egoStateObservation);
  // .def(
  //     py::init<std::string>(),
  //     "Constructor for Scenario",
  //     py::arg("scenarioPath") = "")
}

}  // namespace nocturne
