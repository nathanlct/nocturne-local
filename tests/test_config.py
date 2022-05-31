"""Test configurations passed to the scenario."""
import hydra

from cfgs.config import PROJECT_PATH, get_scenario_dict
from nocturne import Simulation


def test_config_values(cfg):
    """Test that there are no invalid values in the default config."""
    # None in the config would cause a bug
    assert None not in list(get_scenario_dict(cfg).values())


def test_custom_config(cfg):
    """Test that changes in the config are propagated to the scenario."""
    cfg['scenario'].update({
        'max_visible_objects': 3,
        'max_visible_road_points': 14,
        'max_visible_traffic_lights': 15,
        'max_visible_stop_signs': 92,
    })
    scenario_path = str(PROJECT_PATH / 'tests/large_file.json')
    sim = Simulation(scenario_path=scenario_path,
                     config=get_scenario_dict(cfg))
    scenario = sim.getScenario()
    assert scenario.getMaxNumVisibleObjects() == 3
    assert scenario.getMaxNumVisibleRoadPoints() == 14
    assert scenario.getMaxNumVisibleTrafficLights() == 15
    assert scenario.getMaxNumVisibleStopSigns() == 92


@hydra.main(config_path="../cfgs/", config_name="config")
def main(cfg):
    """See file docstring."""
    test_config_values(cfg)
    test_custom_config(cfg)


if __name__ == '__main__':
    main()
