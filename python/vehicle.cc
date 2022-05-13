#include "vehicle.h"

#include <pybind11/pybind11.h>

#include <memory>

#include "geometry/vector_2d.h"
#include "object.h"

namespace py = pybind11;

namespace nocturne {

void DefineVehicle(py::module& m) {
  m.doc() = "nocturne documentation for class Vehicle";

  py::class_<Vehicle, std::shared_ptr<Vehicle>, Object>(m, "Vehicle");
}

}  // namespace nocturne
