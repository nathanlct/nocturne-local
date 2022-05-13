#include "vehicle.h"

#include <pybind11/pybind11.h>

#include <memory>

#include "geometry/vector_2d.h"
#include "object.h"

namespace py = pybind11;

namespace nocturne {

void DefineVehicle(py::module& m) {
  m.doc() = "nocturne documentation for class Vehicle";

  py::class_<Vehicle, std::shared_ptr<Vehicle>, Object>(m, "Vehicle")
      .def_property_readonly("type", &Vehicle::Type);
  // .def_property_readonly("type", [](const Vehicle& vehicle) {
  //   return static_cast<int64_t>(vehicle.Type());
  // });
}

}  // namespace nocturne
