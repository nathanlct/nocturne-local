#include "action.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <limits>
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
      .def_property("steering", &Action::steering, &Action::set_steering)
      .def("numpy",
           [](const Action& action) {
             py::array_t<float> arr(2);
             float* arr_data = arr.mutable_data();
             arr_data[0] = action.acceleration().value_or(
                 std::numeric_limits<float>::quiet_NaN());
             arr_data[1] = action.steering().value_or(
                 std::numeric_limits<float>::quiet_NaN());
             return arr;
           })
      .def_static("from_numpy", [](const py::array_t<float>& arr) {
        assert(arr.size() == 2);
        const float* arr_data = arr.data();
        std::optional<float> acceleration =
            std::isnan(arr_data[0]) ? std::nullopt
                                    : std::make_optional<float>(arr_data[0]);
        std::optional<float> steering =
            std::isnan(arr_data[1]) ? std::nullopt
                                    : std::make_optional<float>(arr_data[1]);
        return Action(acceleration, steering);
      });
  ;
}

}  // namespace nocturne
