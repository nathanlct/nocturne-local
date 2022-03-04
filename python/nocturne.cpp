#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_simulation(py::module&);
void init_object(py::module&);
void init_maths(py::module&);
void init_scenario(py::module&);
void init_roadline(py::module&);

PYBIND11_MODULE(nocturne, m) {
  m.doc() = "Nocturne library - 2D Driving Simulator";

  init_simulation(m);
  init_object(m);
  init_maths(m);
  init_scenario(m);
  init_roadline(m);
}
