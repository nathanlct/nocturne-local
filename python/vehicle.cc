#include "vehicle.h"

#include <pybind11/pybind11.h>

#include <memory>

#include "geometry/vector_2d.h"
#include "kinetic_object.h"

namespace py = pybind11;

namespace nocturne {

void DefineVehicle(py::module& m) {
  m.doc() = "nocturne documentation for class Vehicle";

  py::class_<Vehicle, std::shared_ptr<Vehicle>, KineticObject>(m, "Vehicle")
      .def_property_readonly("type", &Vehicle::Type)
      .def_property_readonly("id", &Vehicle::id)
      .def_property_readonly("length", &Vehicle::length)
      .def_property_readonly("width", &Vehicle::width)
      .def_property(
          "position", &Vehicle::position,
          py::overload_cast<const geometry::Vector2D&>(&Vehicle::set_position))
      .def_property("destination", &Vehicle::destination,
                    py::overload_cast<const geometry::Vector2D&>(
                        &Vehicle::set_destination))
      .def_property("heading", &Vehicle::heading, &Vehicle::set_heading)
      .def_property(
          "velocity", &Vehicle::velocity,
          py::overload_cast<const geometry::Vector2D&>(&Vehicle::set_velocity))
      .def_property("speed", &Vehicle::Speed, &Vehicle::SetSpeed)
      .def_property_readonly("collided", &Vehicle::collided)
      .def_property("acceleration", &Vehicle::acceleration,
                    &Vehicle::set_acceleration)
      .def_property("steering", &Vehicle::steering, &Vehicle::set_steering)
      .def("set_position",
           py::overload_cast<float, float>(&Vehicle::set_position))
      .def("set_destination",
           py::overload_cast<float, float>(&Vehicle::set_destination))
      .def("set_velocity",
           py::overload_cast<float, float>(&Vehicle::set_velocity))

      // TODO: Deprecate the legacy interfaces below.
      .def("getWidth", &Vehicle::width)
      .def("getPosition", &Vehicle::position)
      .def("getGoalPosition", &Vehicle::destination)
      .def("getSpeed", &Vehicle::Speed)
      .def("getHeading", &Vehicle::heading)
      .def("getID", &Vehicle::id)
      .def("getType", &Vehicle::Type)
      .def("getCollided", &Vehicle::collided)
      .def("setAccel", &Vehicle::set_acceleration)
      .def("setSteeringAngle", &Vehicle::set_steering)
      .def("setPosition",
           py::overload_cast<const geometry::Vector2D&>(&Vehicle::set_position))
      .def("setPosition",
           py::overload_cast<float, float>(&Vehicle::set_position))
      .def("setGoalPosition", py::overload_cast<const geometry::Vector2D&>(
                                  &Vehicle::set_destination))
      .def("setGoalPosition",
           py::overload_cast<float, float>(&Vehicle::set_destination))
      .def("setHeading", &Vehicle::set_heading)
      .def("setSpeed", &Vehicle::SetSpeed);
}

}  // namespace nocturne
