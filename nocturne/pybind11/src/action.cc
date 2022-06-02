#include "action.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <cmath>
#include <limits>
#include <string>

#include "nocturne.h"

namespace py = pybind11;

namespace nocturne {

namespace {

py::array_t<float> AsNumpyArray(const Action& action) {
  py::array_t<float> arr(2);
  float* arr_data = arr.mutable_data();
  arr_data[0] =
      action.acceleration().value_or(std::numeric_limits<float>::quiet_NaN());
  arr_data[1] =
      action.steering().value_or(std::numeric_limits<float>::quiet_NaN());
  return arr;
}

Action FromNumpy(const py::array_t<float>& arr) {
  assert(arr.size() == 2);
  const float* arr_data = arr.data();
  std::optional<float> acceleration =
      std::isnan(arr_data[0]) ? std::nullopt
                              : std::make_optional<float>(arr_data[0]);
  std::optional<float> steering = std::isnan(arr_data[1])
                                      ? std::nullopt
                                      : std::make_optional<float>(arr_data[1]);
  return Action(acceleration, steering);
}

}  // namespace

void DefineAction(py::module& m) {
  py::class_<Action>(m, "Action")
      .def(py::init<std::optional<float>, std::optional<float>>(),
           py::arg("acceleration") = py::none(),
           py::arg("steering") = py::none())
      .def("__repr__",
           [](const Action& act) {
             const std::string acceleration_str =
                 act.acceleration().has_value()
                     ? std::to_string(act.acceleration().value())
                     : "None";
             const std::string steering_str =
                 act.steering().has_value()
                     ? std::to_string(act.steering().value())
                     : "None";
             return "{acceleration: " + acceleration_str +
                    ", steering: " + steering_str + "}";
           })
      .def_property("acceleration", &Action::acceleration,
                    &Action::set_acceleration)
      .def_property("steering", &Action::steering, &Action::set_steering)
      .def("numpy", &AsNumpyArray)
      .def_static("from_numpy", &FromNumpy)
      .def(py::pickle(
          [](const Action& act) { return AsNumpyArray(act); },
          [](const py::array_t<float>& arr) { return FromNumpy(arr); }));
}

}  // namespace nocturne
