#include <pybind11/pybind11.h>

#include "ImageMatrix.hpp"
#include "geometry/vector_2d.h"
// #include "../cpp/include/nocturne_bits/ImageMatrix.hpp"
// #include "../cpp/include/nocturne_bits/Vector2D.hpp"
// #include <pybind11/stl.h>  // automatic conversion from C++ std::vector to
// Python list

namespace py = pybind11;

void init_maths(py::module& m) {
  m.doc() = "nocturne documentation for classes Vector2D, ImageMatrix";

  py::class_<nocturne::geometry::Vector2D>(m, "Vector2D")
      // .def_readonly("x", &Vector2D::x)
      // .def_readonly("y", &Vector2D::y);
      .def("x", &nocturne::geometry::Vector2D::x)
      .def("y", &nocturne::geometry::Vector2D::y);

  py::class_<nocturne::ImageMatrix>(m, "ImageMatrix", py::buffer_protocol())
      .def_buffer([](nocturne::ImageMatrix& m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),              /* Pointer to buffer */
            sizeof(unsigned char), /* Size of one scalar */
            py::format_descriptor<unsigned char>::
                format(), /* Python struct-style format descriptor */
            3,            /* Number of dimensions */
            {m.rows(), m.cols(), m.channels()}, /* Buffer dimensions */
            {sizeof(unsigned char) * m.cols() *
                 m.channels(), /* Strides (in bytes) for each index */
             sizeof(unsigned char) * m.channels(), sizeof(unsigned char)});
      });
}
