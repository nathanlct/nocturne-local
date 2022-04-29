#include "road.h"

#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;

namespace nocturne {

void DefineRoadLine(py::module& m) {
  m.doc() = "nocturne documentation for class RoadLine";

  py::class_<RoadLine, std::shared_ptr<RoadLine>>(m, "RoadLine")
      .def_property_readonly("check_collision", &RoadLine::check_collision)
      .def("geometry_points", &RoadLine::geometry_points)

      // TODO: Deprecates the legacy methods below.
      .def("getGeometry", &RoadLine::geometry_points)
      .def("canCollide", &RoadLine::check_collision);
}

}  // namespace nocturne
