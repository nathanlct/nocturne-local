#pragma once

namespace nocturne {

class Action {
 public:
  Action() = default;
  Action(float acceleration, float steering)
      : acceleration_(acceleration), steering_(steering) {}

  float acceleration() const { return acceleration_; }
  void set_acceleration(float acceleration) { acceleration_ = acceleration; }

  float steering() const { return steering_; }
  void set_steering(float steering) { steering_ = steering; }

 protected:
  float acceleration_ = 0.0;
  float steering_ = 0.0f;
};

}  // namespace nocturne
