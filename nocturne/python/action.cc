#include "action.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>

#include "nocturne.h"

namespace py = pybind11;

namespace nocturne {

void DefineAction(py::module& m) {
  py::class_<Action>(m, "Action")
      .def(py::init<float, float>())
      .def("__repr__",
           [](const Action& action) {
             return "{acceleration: " + std::to_string(action.acceleration()) +
                    ", steering: " + std::to_string(action.steering()) + "}";
           })
      .def_property("acceleration", &Action::acceleration,
                    &Action::set_acceleration)
      .def_property("steering", &Action::steering, &Action::set_steering)
      .def("numpy", [](const Action& action) {
        py::array_t<float> arr(2);
        float* arr_data = arr.mutable_data();
        arr_data[0] = action.acceleration();
        arr_data[1] = action.steering();
        return arr;
      });
}

}  // namespace nocturne
