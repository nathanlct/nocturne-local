#include "scenario.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "geometry/geometry_utils.h"
#include "numpy_utils.h"
#include "object.h"

namespace py = pybind11;

namespace nocturne {

using geometry::utils::kHalfPi;

void DefineScenario(py::module& m) {
  m.doc() = "nocturne documentation for class Scenario";

  py::class_<Scenario, std::shared_ptr<Scenario>>(m, "Scenario")
      .def(py::init<const std::string&, int64_t, bool>(),
           "Constructor for Scenario", py::arg("path") = "",
           py::arg("start_time") = 0, py::arg("allow_non_vehicles") = true)

      // Properties
      .def_property_readonly("name", &Scenario::name)
      .def_property_readonly("max_env_time", &Scenario::max_env_time)

      // Methods
      .def("vehicles", &Scenario::vehicles, py::return_value_policy::reference)
      .def("pedestrians", &Scenario::pedestrians,
           py::return_value_policy::reference)
      .def("cyclists", &Scenario::cyclists, py::return_value_policy::reference)
      .def("objects", &Scenario::objects, py::return_value_policy::reference)
      .def("moving_objects", &Scenario::moving_objects,
           py::return_value_policy::reference)
      .def("remove_object", &Scenario::RemoveObject)

      // TODO: Deprecate the legacy interfaces below.
      .def("getVehicles", &Scenario::vehicles,
           py::return_value_policy::reference)
      .def("getPedestrians", &Scenario::pedestrians,
           py::return_value_policy::reference)
      .def("getCyclists", &Scenario::cyclists,
           py::return_value_policy::reference)
      .def("getObjectsThatMoved", &Scenario::moving_objects,
           py::return_value_policy::reference)
      .def("getMaxEnvTime", &Scenario::max_env_time)
      .def("getRoadLines", &Scenario::getRoadLines)
      .def(
          "getCone",
          [](Scenario& scenario, Object* object, float view_dist,
             float view_angle, float head_tilt, bool obscured_view) {
            return utils::AsNumpyArray<unsigned char>(scenario.getCone(
                object, view_dist, view_angle, head_tilt, obscured_view));
          },
          "Draw a cone representing the objects that the agent can see",
          py::arg("object"), py::arg("view_dist") = 60.0,
          py::arg("view_angle") = kHalfPi, py::arg("head_tilt") = 0.0,
          py::arg("obscuredView") = true)
      .def(
          "getImage",
          [](Scenario& scenario, Object* object, bool render_goals) {
            return utils::AsNumpyArray<unsigned char>(
                scenario.getImage(object, render_goals));
          },
          "Return a numpy array of dimension (w, h, 4) representing the scene",
          py::arg("object") = nullptr, py::arg("render_goals") = false)
      .def("removeVehicle", &Scenario::RemoveObject)
      .def("hasExpertAction", &Scenario::hasExpertAction)
      .def("getExpertAction", &Scenario::getExpertAction)
      .def("getExpertSpeeds", &Scenario::getExpertSpeeds)
      .def("getValidExpertStates", &Scenario::getValidExpertStates)
      .def("getMaxNumVisibleObjects", &Scenario::getMaxNumVisibleObjects)
      .def("getMaxNumVisibleRoadPoints", &Scenario::getMaxNumVisibleRoadPoints)
      .def("getMaxNumVisibleStopSigns", &Scenario::getMaxNumVisibleStopSigns)
      .def("getMaxNumVisibleTrafficLights",
           &Scenario::getMaxNumVisibleTrafficLights)
      .def("getObjectFeatureSize", &Scenario::getObjectFeatureSize)
      .def("getRoadPointFeatureSize", &Scenario::getRoadPointFeatureSize)
      .def("getTrafficLightFeatureSize", &Scenario::getTrafficLightFeatureSize)
      .def("getStopSignsFeatureSize", &Scenario::getStopSignsFeatureSize)
      .def("getEgoFeatureSize", &Scenario::getEgoFeatureSize)
      // .def("getVisibleObjects", &Scenario::getVisibleObjects)
      // .def("getVisibleRoadPoints", &Scenario::getVisibleRoadPoints)
      // .def("getVisibleStopSigns", &Scenario::getVisibleStopSigns)
      // .def("getVisibleTrafficLights", &Scenario::getVisibleTrafficLights)
      .def("ego_state",
           [](const Scenario& scenario, const Object& src) {
             return utils::AsNumpyArray(scenario.EgoState(src));
           })
      .def(
          "visible_state",
          [](const Scenario& scenario, const Object& src, float view_dist,
             float view_angle, bool padding) {
            return utils::AsNumpyArrayDict(
                scenario.VisibleState(src, view_dist, view_angle, padding));
          },
          py::arg("object"), py::arg("view_dist") = 60,
          py::arg("view_angle") = kHalfPi, py::arg("padding") = false)
      .def(
          "flattened_visible_state",
          [](const Scenario& scenario, const Object& src, float view_dist,
             float view_angle) {
            return utils::AsNumpyArray(
                scenario.FlattenedVisibleState(src, view_dist, view_angle));
          },
          py::arg("object"), py::arg("view_dist") = 60,
          py::arg("view_angle") = kHalfPi);
}

}  // namespace nocturne
