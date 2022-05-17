#include "object.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cmath>
#include <utility>

#include "geometry/geometry_utils.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace {

using geometry::utils::kHalfPi;
using geometry::utils::kQuarterPi;

constexpr float kTol = 1e-4;

std::pair<geometry::Vector2D, float> KinematicBicycleModel(
    const geometry::Vector2D& position, float length, float heading,
    float speed, float zeta, float dt) {
  const float beta = std::atan(std::tan(zeta) * 0.5f);
  const float dx = speed * std::cos(heading + beta);
  const float dy = speed * std::sin(heading + beta);
  const float dtheta = speed * std::tan(zeta) * std::cos(beta) / length;
  return std::make_pair(position + geometry::Vector2D(dx * dt, dy * dt),
                        geometry::utils::NormalizeAngle(heading + dtheta * dt));
}

TEST(ObjectTest, UniformLinearMotionTest) {
  const float t = 10.0f;
  const float length = 2.0f;
  const float width = 1.0f;
  const float heading = kQuarterPi;
  const float speed = 10.0f;
  const geometry::Vector2D velocity = geometry::PolarToVector2D(speed, heading);
  const geometry::Vector2D position(1.0f, 1.0f);
  const geometry::Vector2D destination = position + velocity * t;

  Object obj(/*id=*/0, length, width, position, destination, heading, speed);
  const int num_steps = 100;
  const float dt = t / static_cast<float>(num_steps);
  for (int i = 0; i < num_steps; ++i) {
    obj.Step(dt);
  }

  EXPECT_FLOAT_EQ(obj.heading(), heading);
  EXPECT_FLOAT_EQ(obj.speed(), speed);
  EXPECT_NEAR(obj.position().x(), destination.x(), kTol);
  EXPECT_NEAR(obj.position().y(), destination.y(), kTol);
}

TEST(ObjectTest, ConstantAccelerationMotionTest) {
  const float t = 10.0f;
  const float length = 2.0f;
  const float width = 1.0f;
  const float heading = kQuarterPi;
  float speed = 0.0f;
  float acceleration = 2.0f;
  geometry::Vector2D velocity = geometry::PolarToVector2D(speed, heading);
  const geometry::Vector2D position(1.0f, 1.0f);
  geometry::Vector2D destination =
      position + velocity * t +
      geometry::PolarToVector2D(acceleration, heading) * (t * t * 0.5f);

  // Forward test.
  Object obj(/*id=*/0, length, width, position, destination, heading, speed);
  obj.set_acceleration(acceleration);
  const int num_steps = 100;
  const float dt = t / static_cast<float>(num_steps);
  for (int i = 0; i < num_steps; ++i) {
    obj.Step(dt);
  }

  EXPECT_FLOAT_EQ(obj.heading(), heading);
  EXPECT_NEAR(obj.speed(), speed + acceleration * t, kTol);
  EXPECT_NEAR(obj.position().x(), destination.x(), kTol);
  EXPECT_NEAR(obj.position().y(), destination.y(), kTol);

  // Backward test.
  speed = 10.0f;
  acceleration = -2.0f;
  velocity = geometry::PolarToVector2D(speed, heading);
  destination =
      position + velocity * t +
      geometry::PolarToVector2D(acceleration, heading) * (t * t * 0.5f);
  obj.set_position(position);
  obj.set_destination(destination);
  obj.set_speed(speed);
  obj.set_acceleration(acceleration);
  for (int i = 0; i < num_steps; ++i) {
    obj.Step(dt);
  }
  EXPECT_FLOAT_EQ(obj.heading(), heading);
  EXPECT_NEAR(obj.speed(), speed + acceleration * t, kTol);
  EXPECT_NEAR(obj.position().x(), destination.x(), kTol);
  EXPECT_NEAR(obj.position().y(), destination.y(), kTol);
}

TEST(ObjectTest, SpeedCliptTest) {
  const float t = 10.0f;
  const float length = 2.0f;
  const float width = 1.0f;
  const float heading = kHalfPi;
  const float max_speed = 10.0f;
  const float speed = 0.0f;
  float acceleration = 2.0f;
  const geometry::Vector2D velocity = geometry::PolarToVector2D(speed, heading);
  geometry::Vector2D final_velocity =
      geometry::PolarToVector2D(max_speed, heading);
  const geometry::Vector2D position(1.0f, 1.0f);
  const float t1 = max_speed / acceleration;
  const float t2 = t - t1;
  geometry::Vector2D destination =
      position + velocity * t1 +
      geometry::PolarToVector2D(acceleration, heading) * (t1 * t1 * 0.5f) +
      final_velocity * t2;

  // Forward test.
  Object obj(/*id=*/0, length, width, max_speed, position, destination, heading,
             speed);
  obj.set_acceleration(acceleration);
  const int num_steps = 100;
  const float dt = t / static_cast<float>(num_steps);
  for (int i = 0; i < num_steps; ++i) {
    obj.Step(dt);
  }

  EXPECT_FLOAT_EQ(obj.heading(), heading);
  EXPECT_NEAR(obj.speed(), max_speed, kTol);
  EXPECT_NEAR(obj.position().x(), destination.x(), kTol);
  EXPECT_NEAR(obj.position().y(), destination.y(), kTol);

  // Backward test.
  acceleration = -2.0f;
  final_velocity = geometry::PolarToVector2D(-max_speed, heading);
  destination =
      position + velocity * t1 +
      geometry::PolarToVector2D(acceleration, heading) * (t1 * t1 * 0.5f) +
      final_velocity * t2;
  obj.set_position(position);
  obj.set_destination(destination);
  obj.set_speed(speed);
  obj.set_acceleration(acceleration);
  for (int i = 0; i < num_steps; ++i) {
    obj.Step(dt);
  }
  EXPECT_FLOAT_EQ(obj.heading(), heading);
  EXPECT_NEAR(obj.speed(), -max_speed, kTol);
  EXPECT_NEAR(obj.position().x(), destination.x(), kTol);
  EXPECT_NEAR(obj.position().y(), destination.y(), kTol);
}

TEST(ObjectTest, SteeringMotionTest) {
  const float length = 2.0f;
  const float width = 1.0f;
  const float heading = kQuarterPi;
  const float speed = 2.0f;
  const float steering = geometry::utils::Radians(10.0f);
  const float dt = 0.1f;
  const geometry::Vector2D position(1.0f, 1.0f);
  const auto [destination, theta] =
      KinematicBicycleModel(position, length, heading, speed, steering, dt);

  Object obj(/*id=*/0, length, width, position, destination, heading, speed);
  obj.set_steering(steering);
  obj.Step(dt);

  EXPECT_FLOAT_EQ(obj.heading(), theta);
  EXPECT_FLOAT_EQ(obj.speed(), speed);
  EXPECT_FLOAT_EQ(obj.position().x(), destination.x());
  EXPECT_FLOAT_EQ(obj.position().y(), destination.y());
  EXPECT_FLOAT_EQ(obj.Velocity().x(), speed * std::cos(theta));
  EXPECT_FLOAT_EQ(obj.Velocity().y(), speed * std::sin(theta));
}

}  // namespace
}  // namespace nocturne
