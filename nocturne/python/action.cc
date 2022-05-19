#include "action.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "nocturne.h"

namespace py = pybind11;

namespace nocturne {

void DefineAction(py::module& m) {
  py::class_<Action>(m, "Action")
      .def(py::init<std::optional<float>, std::optional<float>>(),
           py::arg("acceleration") = py::none(),
           py::arg("steering") = py::none())
      .def("__repr__",
           [](const Action& action) {
             const std::string acceleration_str =
                 action.acceleration().has_value()
                     ? std::to_string(action.acceleration().value())
                     : "None";
             const std::string steering_str =
                 action.steering().has_value()
                     ? std::to_string(action.steering().value())
                     : "None";
             return "{acceleration: " + acceleration_str +
                    ", steering: " + steering_str + "}";
           })
      .def_property("acceleration", &Action::acceleration,
                    &Action::set_acceleration)
      .def_property("steering", &Action::steering, &Action::set_steering);
}

}  // namespace nocturne
