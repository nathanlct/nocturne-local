#include "road.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // automatic conversion from C++ std::vector to

#include <memory>
// Python list

namespace py = pybind11;

void init_roadline(py::module& m) {
  m.doc() = "nocturne documentation for class RoadLine";

  py::class_<nocturne::RoadLine, std::shared_ptr<nocturne::RoadLine>>(
      m, "RoadLine")
      // .def("getRoadPoints", &nocturne::RoadLine::getRoadPoints)
      // .def("getAllPoints", &nocturne::RoadLine::getAllPoints)
      .def("canCollide", &nocturne::RoadLine::check_collision);
  //   .def("getRoadType", &nocturne::RoadLine::getRoadType);
}
