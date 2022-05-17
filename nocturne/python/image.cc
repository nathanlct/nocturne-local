#include "image.h"

#include <pybind11/pybind11.h>

#include "nocturne.h"

namespace py = pybind11;

namespace nocturne {

void DefineImage(py::module& m) {
  m.doc() = "nocturne documentation for class Image";

  py::class_<Image>(m, "Image", py::buffer_protocol())
      .def_buffer([](nocturne::Image& m) -> py::buffer_info {
        return py::buffer_info(
            m.DataPtr(),            // Pointer to buffer
            sizeof(unsigned char),  // Size of one scalar
            py::format_descriptor<unsigned char>::
                format(),  // Python struct-style format descriptor
            3,             // Number of dimensions
            {m.rows(), m.cols(), m.channels()},  // Buffer dimensions
            {sizeof(unsigned char) * m.cols() *
                 m.channels(),  // Strides (in bytes) for each index
             sizeof(unsigned char) * m.channels(), sizeof(unsigned char)});
      });
}

}  // namespace nocturne
