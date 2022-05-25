"""Test configurations passed to the scenario."""
from cfgs.config import PROJECT_PATH, get_default_config
from nocturne import Simulation


def test_config_values():
    """Test that there are no invalid values in the default config."""
    config = get_default_config()

    # None in the config would cause a bug
    assert None not in list(config.values())


def test_custom_config():
    """Test that changes in the config are propagated to the scenario."""
    config = get_default_config({
        'max_visible_objects': 3,
        'max_visible_road_points': 14,
        'max_visible_traffic_lights': 15,
        'max_visible_stop_signs': 92,
    })
    scenario_path = str(PROJECT_PATH / 'tests/large_file.json')
    sim = Simulation(scenario_path=scenario_path, config=config)
    scenario = sim.getScenario()
    assert scenario.getMaxNumVisibleObjects() == 3
    assert scenario.getMaxNumVisibleRoadPoints() == 14
    assert scenario.getMaxNumVisibleTrafficLights() == 15
    assert scenario.getMaxNumVisibleStopSigns() == 92


if __name__ == '__main__':
    test_config_values()
    test_custom_config()
