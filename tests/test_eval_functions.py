"""Test evaluation functions."""
from nocturne.utils.eval.average_displacement import compute_average_displacement
from nocturne.utils.eval.goal_reaching_rate import compute_average_goal_reaching_rate
from nocturne.utils.eval.collision_rate import compute_average_collision_rate

import numpy as np


def test_eval_functions():
    """Test evaluation functions."""
    trajectory_path = ['tests/large_file.json']

    # average displacement
    np.testing.assert_allclose(
        compute_average_displacement(
            trajectory_path,
            model=lambda _: [[1.0, 3.14 / 8.0]],
            sim_allow_non_vehicles=True),
        21.337524
    )

    # collision rate
    np.testing.assert_allclose(
        compute_average_collision_rate(
            trajectory_path,
            model=None,
            sim_allow_non_vehicles=True,
            check_vehicles_only=True),
        np.array([0.0, 0.0])
    )

    np.testing.assert_allclose(
        compute_average_collision_rate(
            trajectory_path,
            model=None,
            sim_allow_non_vehicles=True,
            check_vehicles_only=False),
        np.array([0.18181818, 0.09090909])
    )

    # goal reaching rate
    np.testing.assert_allclose(
        compute_average_goal_reaching_rate(
            trajectory_path,
            model=None,
            sim_allow_non_vehicles=True,
            check_vehicles_only=False),
        1.0
    )


if __name__ == '__main__':
    test_eval_functions()
