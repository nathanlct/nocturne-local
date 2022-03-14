#include "Object.hpp"

#include <pybind11/pybind11.h>

#include <memory>

#include "Vehicle.hpp"
// #include <pybind11/stl.h>  // automatic conversion from C++ std::vector to
// Python list

namespace py = pybind11;

// TODO: Use property to replace methods here.
void init_object(py::module& m) {
  m.doc() = "nocturne documentation for class Object and subclass Vehicle";

  py::class_<nocturne::Object, std::shared_ptr<nocturne::Object>>(m, "Object")
      .def("getWidth", &nocturne::Object::width)
      .def("getLength", &nocturne::Object::length)
      .def("getPosition", &nocturne::Object::position)
      .def("getGoalPosition", &nocturne::Object::goal_position)
      .def("getSpeed", &nocturne::Object::speed)
      .def("getHeading", &nocturne::Object::heading)
      .def("getID", &nocturne::Object::id)
      .def("getType", &nocturne::Object::Type)
      .def("getCollided", &nocturne::Object::collided)
      .def("setPosition", &nocturne::Object::set_position)
      .def("setGoalPosition", &nocturne::Object::set_goal_position)
      .def("setSpeed", &nocturne::Object::set_speed)
      .def("setHeading", &nocturne::Object::set_heading);

  py::class_<nocturne::Vehicle, std::shared_ptr<nocturne::Vehicle>,
             nocturne::Object>(m, "Vehicle")
      .def("getWidth", &nocturne::Vehicle::width)
      .def("getPosition", &nocturne::Vehicle::position)
      .def("getGoalPosition", &nocturne::Vehicle::goal_position)
      .def("getSpeed", &nocturne::Vehicle::speed)
      .def("getHeading", &nocturne::Vehicle::heading)
      .def("getID", &nocturne::Vehicle::id)
      .def("getType", &nocturne::Vehicle::Type)
      .def("getCollided", &nocturne::Vehicle::collided)
      .def("setAccel", &nocturne::Vehicle::setAccel)
      .def("setSteeringAngle", &nocturne::Vehicle::setSteeringAngle)
      .def("setPosition", &nocturne::Vehicle::set_position)
      .def("setGoalPosition", &nocturne::Vehicle::set_goal_position)
      .def("setSpeed", &nocturne::Vehicle::set_speed)
      .def("setHeading", &nocturne::Vehicle::set_heading);
}
