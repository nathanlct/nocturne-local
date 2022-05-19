#pragma once

#include <optional>

namespace nocturne {

class Action {
 public:
  Action() = default;
  Action(std::optional<float> acceleration, std::optional<float> steering)
      : acceleration_(acceleration), steering_(steering) {}

  std::optional<float> acceleration() const { return acceleration_; }
  void set_acceleration(std::optional<float> acceleration) {
    acceleration_ = acceleration;
  }

  std::optional<float> steering() const { return steering_; }
  void set_steering(std::optional<float> steering) { steering_ = steering; }

 protected:
  std::optional<float> acceleration_ = std::nullopt;
  std::optional<float> steering_ = std::nullopt;
};

}  // namespace nocturne
