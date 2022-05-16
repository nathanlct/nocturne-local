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
      .def(
          "getImage",
          [](Scenario& scenario, uint64_t img_width, uint64_t img_height,
             bool draw_destinations, float padding, Object* source,
             uint64_t view_width, uint64_t view_height,
             bool rotate_with_source) {
            return utils::AsNumpyArray<unsigned char>(scenario.Image(
                img_width, img_height, draw_destinations, padding, source,
                view_width, view_height, rotate_with_source));
          },
          "Return a numpy array of dimension (img_height, img_width, 4) "
          "representing an image of the scene.",
          py::arg("img_width") = 1000, py::arg("img_height") = 1000,
          py::arg("draw_destinations") = true, py::arg("padding") = 50.0f,
          py::arg("source") = nullptr, py::arg("view_width") = 200,
          py::arg("view_height") = 200, py::arg("rotate_with_source") = true)
      .def(
          "getFeaturesImage",
          [](Scenario& scenario, const Object& source, float view_dist,
             float view_angle, float head_tilt, uint64_t img_width,
             uint64_t img_height, float padding, bool draw_destination) {
            return utils::AsNumpyArray<unsigned char>(
                scenario.EgoVehicleFeaturesImage(
                    source, view_dist, view_angle, head_tilt, img_width,
                    img_height, padding, draw_destination));
          },
          "Return a numpy array of dimension (img_height, img_width, 4) "
          "representing an image of what is returned by getVisibleState(?).",
          py::arg("source"), py::arg("view_dist") = 120.0f,
          py::arg("view_angle") = geometry::utils::kPi * 0.8f,
          py::arg("head_tilt") = 0.0f, py::arg("img_width") = 1000,
          py::arg("img_height") = 1000, py::arg("padding") = 0.0f,
          py::arg("draw_destination") = true)
      .def(
          "getConeImage",
          [](Scenario& scenario, const Object& source, float view_dist,
             float view_angle, float head_tilt, uint64_t img_width,
             uint64_t img_height, float padding, bool draw_destination) {
            return utils::AsNumpyArray<unsigned char>(
                scenario.EgoVehicleConeImage(source, view_dist, view_angle,
                                             head_tilt, img_width, img_height,
                                             padding, draw_destination));
          },
          "Return a numpy array of dimension (img_height, img_width, 4) "
          "representing a cone of what the agent sees.",
          py::arg("source"), py::arg("view_dist") = 120.0f,
          py::arg("view_angle") = geometry::utils::kPi * 0.8f,
          py::arg("head_tilt") = 0.0f, py::arg("img_width") = 1000,
          py::arg("img_height") = 1000, py::arg("padding") = 0.0f,
          py::arg("draw_destination") = true)
      .def("removeVehicle", &Scenario::removeVehicle)
      .def("hasExpertAction", &Scenario::hasExpertAction)
      .def("getExpertAction", &Scenario::getExpertAction)
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
