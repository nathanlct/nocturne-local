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
          py::arg("view_angle") = kHalfPi)
      .def("expert_heading", &Scenario::ExpertHeading)
      .def("expert_speed", &Scenario::ExpertSpeed)
      .def("expert_velocity", &Scenario::ExpertVelocity)
      .def("expert_action", &Scenario::ExpertAction)

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
          "getImage",
          [](Scenario& scenario, uint64_t img_height, uint64_t img_width,
             bool draw_destinations, float padding, Object* source,
             uint64_t view_height, uint64_t view_width,
             bool rotate_with_source) {
            return utils::AsNumpyArray<unsigned char>(scenario.Image(
                img_height, img_width, draw_destinations, padding, source,
                view_height, view_width, rotate_with_source));
          },
          "Return a numpy array of dimension (img_height, img_width, 4) "
          "representing an image of the scene.",
          py::arg("img_height") = 1000, py::arg("img_width") = 1000,
          py::arg("draw_destinations") = true, py::arg("padding") = 50.0f,
          py::arg("source") = nullptr, py::arg("view_height") = 200,
          py::arg("view_width") = 200, py::arg("rotate_with_source") = true)
      .def(
          "getFeaturesImage",
          [](Scenario& scenario, const Object& source, float view_dist,
             float view_angle, float head_tilt, uint64_t img_height,
             uint64_t img_width, float padding, bool draw_destination) {
            return utils::AsNumpyArray<unsigned char>(
                scenario.EgoVehicleFeaturesImage(
                    source, view_dist, view_angle, head_tilt, img_height,
                    img_width, padding, draw_destination));
          },
          "Return a numpy array of dimension (img_height, img_width, 4) "
          "representing an image of what is returned by getVisibleState(?).",
          py::arg("source"), py::arg("view_dist") = 120.0f,
          py::arg("view_angle") = geometry::utils::kPi * 0.8f,
          py::arg("head_tilt") = 0.0f, py::arg("img_height") = 1000,
          py::arg("img_width") = 1000, py::arg("padding") = 0.0f,
          py::arg("draw_destination") = true)
      .def(
          "getConeImage",
          [](Scenario& scenario, const Object& source, float view_dist,
             float view_angle, float head_tilt, uint64_t img_height,
             uint64_t img_width, float padding, bool draw_destination) {
            return utils::AsNumpyArray<unsigned char>(
                scenario.EgoVehicleConeImage(source, view_dist, view_angle,
                                             head_tilt, img_height, img_width,
                                             padding, draw_destination));
          },
          "Return a numpy array of dimension (img_height, img_width, 4) "
          "representing a cone of what the agent sees.",
          py::arg("source"), py::arg("view_dist") = 120.0f,
          py::arg("view_angle") = geometry::utils::kPi * 0.8f,
          py::arg("head_tilt") = 0.0f, py::arg("img_height") = 1000,
          py::arg("img_width") = 1000, py::arg("padding") = 0.0f,
          py::arg("draw_destination") = true)
      .def("removeVehicle", &Scenario::RemoveObject)
      // .def("hasExpertAction", &Scenario::hasExpertAction)
      .def("getExpertAction", &Scenario::ExpertAction)
      .def("getExpertSpeeds", &Scenario::ExpertVelocity)
      .def("getMaxNumVisibleObjects", &Scenario::getMaxNumVisibleObjects)
      .def("getMaxNumVisibleRoadPoints", &Scenario::getMaxNumVisibleRoadPoints)
      .def("getMaxNumVisibleStopSigns", &Scenario::getMaxNumVisibleStopSigns)
      .def("getMaxNumVisibleTrafficLights",
           &Scenario::getMaxNumVisibleTrafficLights)
      .def("getObjectFeatureSize", &Scenario::getObjectFeatureSize)
      .def("getRoadPointFeatureSize", &Scenario::getRoadPointFeatureSize)
      .def("getTrafficLightFeatureSize", &Scenario::getTrafficLightFeatureSize)
      .def("getStopSignsFeatureSize", &Scenario::getStopSignsFeatureSize)
      .def("getEgoFeatureSize", &Scenario::getEgoFeatureSize);
}

}  // namespace nocturne
