#include "object.h"

#include <pybind11/pybind11.h>

#include <memory>

#include "Vehicle.hpp"
#include "geometry/vector_2d.h"
// #include <pybind11/stl.h>  // automatic conversion from C++ std::vector to
// Python list

namespace py = pybind11;

// TODO: Use property to replace methods here.
void init_object(py::module& m) {
  m.doc() = "nocturne documentation for class Object and subclass Vehicle";

  py::class_<nocturne::MovingObject, std::shared_ptr<nocturne::MovingObject>>(
      m, "Object")
      .def("getWidth", &nocturne::MovingObject::width)
      .def("getLength", &nocturne::MovingObject::length)
      .def("getPosition", &nocturne::MovingObject::position)
      .def("getGoalPosition", &nocturne::MovingObject::destination)
      .def("getSpeed", &nocturne::MovingObject::speed)
      .def("getHeading", &nocturne::MovingObject::heading)
      .def("getID", &nocturne::MovingObject::id)
      .def("getType", &nocturne::MovingObject::Type)
      .def("getCollided", &nocturne::MovingObject::collided)
      .def("setPosition",
           py::overload_cast<const nocturne::geometry::Vector2D&>(
               &nocturne::MovingObject::set_position))
      .def("setPosition", py::overload_cast<float, float>(
                              &nocturne::MovingObject::set_position))
      .def("setGoalPosition",
           py::overload_cast<const nocturne::geometry::Vector2D&>(
               &nocturne::MovingObject::set_destination))
      .def("setGoalPosition", py::overload_cast<float, float>(
                                  &nocturne::MovingObject::set_destination))
      .def("setSpeed", &nocturne::MovingObject::set_speed)
      .def("setHeading", &nocturne::MovingObject::set_heading);

  py::class_<nocturne::Vehicle, std::shared_ptr<nocturne::Vehicle>,
             nocturne::Object>(m, "Vehicle")
      .def("getWidth", &nocturne::Vehicle::width)
      .def("getPosition", &nocturne::Vehicle::position)
      .def("getGoalPosition", &nocturne::Vehicle::destination)
      .def("getSpeed", &nocturne::Vehicle::speed)
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
      .def("setSpeed", &nocturne::Vehicle::set_speed)
      .def("setHeading", &nocturne::Vehicle::set_heading);
}
