#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_simulation(py::module &);

PYBIND11_MODULE(nocturne, m) {
    m.doc() = "Nocturne library - 2D Driving Simulator";
    
    init_simulation(m);
}