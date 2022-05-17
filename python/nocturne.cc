#include "nocturne.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

PYBIND11_MODULE(nocturne, m) {
  m.doc() = "Nocturne library - 2D Driving Simulator";

  nocturne::DefineObject(m);
  nocturne::DefineRoadLine(m);
  nocturne::DefineScenario(m);
  nocturne::DefineSimulation(m);
  nocturne::DefineVector2D(m);
  nocturne::DefineVehicle(m);
}

}  // namespace
