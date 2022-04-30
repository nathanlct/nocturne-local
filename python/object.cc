#include "object.h"

#include <pybind11/pybind11.h>

#include <memory>

#include "geometry/vector_2d.h"

namespace py = pybind11;

namespace nocturne {

void DefineObject(py::module& m) {
  m.doc() = "nocturne documentation for class Object";

  py::class_<Object, std::shared_ptr<Object>>(m, "Object")
      .def_property_readonly(
          "type",
          [](const Object& obj) { return static_cast<int64_t>(obj.Type()); })
      .def_property_readonly("id", &Object::id)
      .def_property_readonly("length", &Object::length)
      .def_property_readonly("width", &Object::width)
      .def_property(
          "position", &Object::position,
          py::overload_cast<const geometry::Vector2D&>(&Object::set_position))
      .def_property("destination", &Object::destination,
                    py::overload_cast<const geometry::Vector2D&>(
                        &Object::set_destination))
      .def_property("heading", &Object::heading, &Object::set_heading)
      .def_property(
          "velocity", &Object::velocity,
          py::overload_cast<const geometry::Vector2D&>(&Object::set_velocity))
      .def_property("speed", &Object::Speed, &Object::SetSpeed)
      .def_property("manual_control", &Object::manual_control,
                    &Object::set_manual_control)
      .def_property("expert_control", &Object::expert_control,
                    &Object::set_expert_control)
      .def_property_readonly("collided", &Object::collided)
      .def("set_position",
           py::overload_cast<float, float>(&Object::set_position))
      .def("set_destination",
           py::overload_cast<float, float>(&Object::set_destination))
      .def("set_velocity",
           py::overload_cast<float, float>(&Object::set_velocity))

      // TODO: Deprecate the legacy interfaces below.
      .def("getWidth", &Object::width)
      .def("getLength", &Object::length)
      .def("getPosition", &Object::position)
      .def("getGoalPosition", &Object::destination)
      .def("getSpeed", &Object::Speed)
      .def("getHeading", &Object::heading)
      .def("getID", &Object::id)
      .def("getType", &Object::Type)
      .def("getCollided", &Object::collided)
      .def("setPosition",
           py::overload_cast<const geometry::Vector2D&>(&Object::set_position))
      .def("setPosition",
           py::overload_cast<float, float>(&Object::set_position))
      .def("setGoalPosition", py::overload_cast<const geometry::Vector2D&>(
                                  &Object::set_destination))
      .def("setGoalPosition",
           py::overload_cast<float, float>(&Object::set_destination))
      .def("setHeading", &Object::set_heading)
      .def("setSpeed", &Object::SetSpeed);
}

}  // namespace nocturne
