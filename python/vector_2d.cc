#include "geometry/vector_2d.h"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
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
      .def(py::init<float, float>())
      .def_property("x", &geometry::Vector2D::x, &geometry::Vector2D::set_x)
      .def_property("y", &geometry::Vector2D::y, &geometry::Vector2D::set_y)
      // Operators
      .def(-py::self)
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self + float())
      .def(float() + py::self)
      .def(py::self += float())
      .def(py::self - py::self)
      .def(py::self -= py::self)
      .def(py::self - float())
      .def(py::self -= float())
      .def(py::self * float())
      .def(float() * py::self)
      .def(py::self *= float())
      .def(py::self / float())
      .def(py::self /= float())
      // Methods
      .def("norm", &geometry::Vector2D::Norm, py::arg("p") = 2)
      .def("angle", &geometry::Vector2D::Angle)
      .def("rotate", &geometry::Vector2D::Rotate)
      .def("numpy", [](const geometry::Vector2D& vec) {
        py::array_t<float> arr(2);
        float* arr_data = arr.mutable_data();
        arr_data[0] = vec.x();
        arr_data[1] = vec.y();
        return arr;
      });
}

}  // namespace nocturne
