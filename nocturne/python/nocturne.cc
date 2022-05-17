#include "nocturne.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nocturne {
namespace {

PYBIND11_MODULE(nocturne_cpp, m) {
  m.doc() = "Nocturne library - 2D Driving Simulator";

  DefineImage(m);
  DefineObject(m);
  DefineRoadLine(m);
  DefineScenario(m);
  DefineSimulation(m);
  DefineVector2D(m);
  DefineVehicle(m);
}

}  // namespace
}  // namespace nocturne
