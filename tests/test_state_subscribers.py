from nocturne import Simulation
sim = Simulation(scenario_path='/checkpoint/eugenevinitsky/waymo_open/motion_v1p1/uncompressed/scenario/formatted_json/tfrecord-00002-of-01000_5.json')
scenario = sim.getScenario()
vehs = scenario.getVehicles()
state = scenario.getEgoState(vehs[0])
import ipdb; ipdb.set_trace()
state = scenario.getVisibleObjectsState(vehs[0])