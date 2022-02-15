#include <pybind11/pybind11.h>

#include <memory>

#include "Object.hpp"
#include "Vehicle.hpp"
// #include <pybind11/stl.h>  // automatic conversion from C++ std::vector to
// Python list

namespace py = pybind11;

void init_object(py::module& m) {
  m.doc() = "nocturne documentation for class Object and subclass Vehicle";

  py::class_<nocturne::Object, std::shared_ptr<nocturne::Object>>(m, "Object")
      .def("getWidth", &nocturne::Object::getWidth)
      .def("getLength", &nocturne::Object::getLength)
      .def("getPosition", &nocturne::Object::getPosition)
      .def("getGoalPosition", &nocturne::Object::getGoalPosition)
      .def("getSpeed", &nocturne::Object::getSpeed)
      .def("getHeading", &nocturne::Object::getHeading)
      .def("getID", &nocturne::Object::getID)
      .def("getType", &nocturne::Object::getType)
      .def("getCollided", &nocturne::Object::getCollided)
      .def("setPosition", &nocturne::Object::setPosition)
      .def("setGoalPosition", &nocturne::Object::setGoalPosition)
      .def("setSpeed", &nocturne::Object::setSpeed)
      .def("setHeading", &nocturne::Object::setHeading);

  py::class_<nocturne::Vehicle, std::shared_ptr<nocturne::Vehicle>,
             nocturne::Object>(m, "Vehicle")
      .def("getWidth", &nocturne::Vehicle::getWidth)
      .def("getPosition", &nocturne::Vehicle::getPosition)
      .def("getGoalPosition", &nocturne::Vehicle::getGoalPosition)
      .def("getSpeed", &nocturne::Vehicle::getSpeed)
      .def("getHeading", &nocturne::Vehicle::getHeading)
      .def("getID", &nocturne::Vehicle::getID)
      .def("getType", &nocturne::Vehicle::getType)
      .def("getCollided", &nocturne::Vehicle::getCollided)
      .def("setAccel", &nocturne::Vehicle::setAccel)
      .def("setSteeringAngle", &nocturne::Vehicle::setSteeringAngle)
      .def("setPosition", &nocturne::Vehicle::setPosition)
      .def("setGoalPosition", &nocturne::Vehicle::setGoalPosition)
      .def("setSpeed", &nocturne::Vehicle::setSpeed)
      .def("setHeading", &nocturne::Vehicle::setHeading);
}
