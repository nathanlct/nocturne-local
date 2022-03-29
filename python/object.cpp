#include "object.h"

#include <pybind11/pybind11.h>

#include <memory>

#include "geometry/vector_2d.h"
#include "vehicle.h"
// #include <pybind11/stl.h>  // automatic conversion from C++ std::vector to
// Python list

namespace py = pybind11;

// TODO: Use property to replace methods here.
void init_object(py::module& m) {
  m.doc() = "nocturne documentation for class Object and subclass Vehicle";

  py::class_<nocturne::KineticObject, std::shared_ptr<nocturne::KineticObject>>(
      m, "Object")
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
      .def("set_velocity",
           py::overload_cast<const nocturne::geometry::Vector2D&>(
               &nocturne::KineticObject::set_velocity))
      .def("set_velocity", py::overload_cast<float, float>(
                               &nocturne::KineticObject::set_velocity))
      .def("setSpeed", &nocturne::KineticObject::SetSpeed);

  py::class_<nocturne::Vehicle, std::shared_ptr<nocturne::Vehicle>,
             nocturne::KineticObject>(m, "Vehicle")
      .def("getWidth", &nocturne::Vehicle::width)
      .def("getPosition", &nocturne::Vehicle::position)
      .def("getGoalPosition", &nocturne::Vehicle::destination)
      .def("getSpeed", &nocturne::Vehicle::Speed)
      .def("getHeading", &nocturne::Vehicle::heading)
      .def("getID", &nocturne::Vehicle::id)
      .def("getType", &nocturne::Vehicle::Type)
      .def("getCollided", &nocturne::Vehicle::collided)
      .def("setAccel", &nocturne::Vehicle::setAccel)
      .def("setSteeringAngle", &nocturne::Vehicle::setSteeringAngle)
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
      .def("set_velocity",
           py::overload_cast<const nocturne::geometry::Vector2D&>(
               &nocturne::KineticObject::set_velocity))
      .def("set_velocity", py::overload_cast<float, float>(
                               &nocturne::KineticObject::set_velocity))
      .def("setSpeed", &nocturne::Vehicle::SetSpeed);
}
