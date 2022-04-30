#include "geometry/vector_2d.h"

#include <pybind11/pybind11.h>

#include <string>

#include "nocturne.h"

namespace py = pybind11;

namespace nocturne {

void DefineVector2D(py::module& m) {
  m.doc() = "nocturne documentation for class Vector2D";

  py::class_<geometry::Vector2D>(m, "Vector2D")
      .def("__repr__",
           [](const geometry::Vector2D& vec) {
             return "(" + std::to_string(vec.x()) + ", " +
                    std::to_string(vec.y()) + ")";
           })
      .def_property("x", &nocturne::geometry::Vector2D::x,
                    &nocturne::geometry::Vector2D::set_x)
      .def_property("y", &nocturne::geometry::Vector2D::y,
                    &nocturne::geometry::Vector2D::set_y);
}

}  // namespace nocturne
