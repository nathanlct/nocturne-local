#include "kinetic_object.h"

#include <pybind11/pybind11.h>

#include <memory>

#include "geometry/vector_2d.h"

namespace py = pybind11;

namespace nocturne {

// TODO: Use property to replace methods here.
void DefineKineticObject(py::module& m) {
  m.doc() = "nocturne documentation for class KineticObject";

  py::class_<KineticObject, std::shared_ptr<KineticObject>>(m, "KineticObject")
      .def_property_readonly("type", &KineticObject::Type)
      .def_property_readonly("id", &KineticObject::id)
      .def_property_readonly("length", &KineticObject::length)
      .def_property_readonly("width", &KineticObject::width)
      .def_property("position", &KineticObject::position,
                    py::overload_cast<const geometry::Vector2D&>(
                        &KineticObject::set_position))
      .def_property("destination", &KineticObject::destination,
                    py::overload_cast<const geometry::Vector2D&>(
                        &KineticObject::set_destination))
      .def_property("heading", &KineticObject::heading,
                    &KineticObject::set_heading)
      .def_property("velocity", &KineticObject::velocity,
                    py::overload_cast<const geometry::Vector2D&>(
                        &KineticObject::set_velocity))
      .def_property("speed", &KineticObject::Speed, &KineticObject::SetSpeed)
      .def_property("keyboard_controllable",
                    &KineticObject::keyboard_controllable,
                    &KineticObject::set_keyboard_controllable)
      .def_property_readonly("collided", &KineticObject::collided)
      .def("set_position",
           py::overload_cast<float, float>(&KineticObject::set_position))
      .def("set_destination",
           py::overload_cast<float, float>(&KineticObject::set_destination))
      .def("set_velocity",
           py::overload_cast<float, float>(&KineticObject::set_velocity))
      .def("set_expert_controlled",
           py::overload_cast<bool>(&KineticObject::set_expert_controlled))

      // TODO: Deprecate the legacy interfaces below.
      .def("getWidth", &KineticObject::width)
      .def("getLength", &KineticObject::length)
      .def("getPosition", &KineticObject::position)
      .def("getGoalPosition", &KineticObject::destination)
      .def("getSpeed", &KineticObject::Speed)
      .def("getHeading", &KineticObject::heading)
      .def("getID", &KineticObject::id)
      .def("getType", &KineticObject::Type)
      .def("getCollided", &KineticObject::collided)
      .def("setPosition", py::overload_cast<const geometry::Vector2D&>(
                              &KineticObject::set_position))
      .def("setPosition",
           py::overload_cast<float, float>(&KineticObject::set_position))
      .def("setGoalPosition", py::overload_cast<const geometry::Vector2D&>(
                                  &KineticObject::set_destination))
      .def("setGoalPosition",
           py::overload_cast<float, float>(&KineticObject::set_destination))
      .def("setHeading", &KineticObject::set_heading)
      .def("setSpeed", &KineticObject::SetSpeed);
}

}  // namespace nocturne
