#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nocturne {

void DefineObject(py::module& m);
void DefineRoadLine(py::module& m);
void DefineScenario(py::module& m);
void DefineSimulation(py::module& m);
void DefineVector2D(py::module& m);
void DefineVehicle(py::module& m);

}  // namespace nocturne
