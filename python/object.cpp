#include "object.h"

#include <pybind11/pybind11.h>

#include <memory>

#include "geometry/vector_2d.h"
#include "kinetic_object.h"
#include "vehicle.h"
// #include <pybind11/stl.h>  // automatic conversion from C++ std::vector to
// Python list

namespace py = pybind11;

// TODO: Use property to replace methods here.
void init_object(py::module& m) {
  m.doc() = "nocturne documentation for class Object and subclass Vehicle";

  py::class_<nocturne::KineticObject, std::shared_ptr<nocturne::KineticObject>>(
      m, "Object")
      .def_property_readonly("type", &nocturne::KineticObject::Type)
      .def_property_readonly("id", &nocturne::KineticObject::id)
      .def_property_readonly("length", &nocturne::KineticObject::length)
      .def_property_readonly("width", &nocturne::KineticObject::width)
      .def_property("position", &nocturne::KineticObject::position,
                    py::overload_cast<const nocturne::geometry::Vector2D&>(
                        &nocturne::KineticObject::set_position))
      .def_property("destination", &nocturne::KineticObject::destination,
                    py::overload_cast<const nocturne::geometry::Vector2D&>(
                        &nocturne::KineticObject::set_destination))
      .def_property("heading", &nocturne::KineticObject::heading,
                    &nocturne::KineticObject::set_heading)
      .def_property("velocity", &nocturne::KineticObject::velocity,
                    py::overload_cast<const nocturne::geometry::Vector2D&>(
                        &nocturne::KineticObject::set_velocity))
      .def_property("speed", &nocturne::KineticObject::Speed,
                    &nocturne::KineticObject::SetSpeed)
      .def_property_readonly("collided", &nocturne::KineticObject::collided)
      .def("set_position", py::overload_cast<float, float>(
                               &nocturne::KineticObject::set_position))
      .def("set_destination", py::overload_cast<float, float>(
                                  &nocturne::KineticObject::set_destination))
      .def("set_velocity", py::overload_cast<float, float>(
                               &nocturne::KineticObject::set_velocity))

      // TODO: Deprecate the legacy interfaces below.
      .def("getWidth", &nocturne::KineticObject::width)
      .def("getLength", &nocturne::KineticObject::length)
      .def("getPosition", &nocturne::KineticObject::position)
      .def("getGoalPosition", &nocturne::KineticObject::destination)
      .def("getSpeed", &nocturne::KineticObject::Speed)
      .def("getHeading", &nocturne::KineticObject::heading)
      .def("getID", &nocturne::KineticObject::id)
      .def("getType", &nocturne::KineticObject::Type)
      .def("getCollided", &nocturne::KineticObject::collided)
      .def("setPosition",
           py::overload_cast<const nocturne::geometry::Vector2D&>(
               &nocturne::KineticObject::set_position))
      .def("setPosition", py::overload_cast<float, float>(
                              &nocturne::KineticObject::set_position))
      .def("setGoalPosition",
           py::overload_cast<const nocturne::geometry::Vector2D&>(
               &nocturne::KineticObject::set_destination))
      .def("setGoalPosition", py::overload_cast<float, float>(
                                  &nocturne::KineticObject::set_destination))
      .def("setHeading", &nocturne::KineticObject::set_heading)
      .def("setSpeed", &nocturne::KineticObject::SetSpeed);

  py::class_<nocturne::Vehicle, std::shared_ptr<nocturne::Vehicle>,
             nocturne::KineticObject>(m, "Vehicle")
      .def_property_readonly("type", &nocturne::Vehicle::Type)
      .def_property_readonly("id", &nocturne::Vehicle::id)
      .def_property_readonly("length", &nocturne::Vehicle::length)
      .def_property_readonly("width", &nocturne::Vehicle::width)
      .def_property("position", &nocturne::Vehicle::position,
                    py::overload_cast<const nocturne::geometry::Vector2D&>(
                        &nocturne::Vehicle::set_position))
      .def_property("destination", &nocturne::Vehicle::destination,
                    py::overload_cast<const nocturne::geometry::Vector2D&>(
                        &nocturne::Vehicle::set_destination))
      .def_property("heading", &nocturne::Vehicle::heading,
                    &nocturne::Vehicle::set_heading)
      .def_property("velocity", &nocturne::Vehicle::velocity,
                    py::overload_cast<const nocturne::geometry::Vector2D&>(
                        &nocturne::Vehicle::set_velocity))
      .def_property("speed", &nocturne::Vehicle::Speed,
                    &nocturne::Vehicle::SetSpeed)
      .def_property_readonly("collided", &nocturne::Vehicle::collided)
      .def_property("acceleration", &nocturne::Vehicle::acceleration,
                    &nocturne::Vehicle::set_acceleration)
      .def_property("steering", &nocturne::Vehicle::steering,
                    &nocturne::Vehicle::set_steering)
      .def("set_position",
           py::overload_cast<float, float>(&nocturne::Vehicle::set_position))
      .def("set_destination",
           py::overload_cast<float, float>(&nocturne::Vehicle::set_destination))
      .def("set_velocity",
           py::overload_cast<float, float>(&nocturne::Vehicle::set_velocity))

      // TODO: Deprecate the legacy interfaces below.
      .def("getWidth", &nocturne::Vehicle::width)
      .def("getPosition", &nocturne::Vehicle::position)
      .def("getGoalPosition", &nocturne::Vehicle::destination)
      .def("getSpeed", &nocturne::Vehicle::Speed)
      .def("getHeading", &nocturne::Vehicle::heading)
      .def("getID", &nocturne::Vehicle::id)
      .def("getType", &nocturne::Vehicle::Type)
      .def("getCollided", &nocturne::Vehicle::collided)
      .def("setAccel", &nocturne::Vehicle::set_acceleration)
      .def("setSteeringAngle", &nocturne::Vehicle::set_steering)
      .def("setPosition",
           py::overload_cast<const nocturne::geometry::Vector2D&>(
               &nocturne::Vehicle::set_position))
      .def("setPosition",
           py::overload_cast<float, float>(&nocturne::Vehicle::set_position))
      .def("setGoalPosition",
           py::overload_cast<const nocturne::geometry::Vector2D&>(
               &nocturne::Vehicle::set_destination))
      .def("setGoalPosition",
           py::overload_cast<float, float>(&nocturne::Vehicle::set_destination))
      .def("setHeading", &nocturne::Vehicle::set_heading)
      .def("setSpeed", &nocturne::Vehicle::SetSpeed);
}
