import numpy as np
from nocturne import Simulation
import matplotlib.pyplot as plt
import time

sim = Simulation(scenarioPath='./scenarios/intersection.json')
scenario = sim.getScenario()

sim.step(0.1)

objects = scenario.getRoadObjects()
print([o.getID() for o in objects])
print([o.getType() for o in objects])

vehicles = scenario.getVehicles()
print([(v.getID(), v.getType()) for v in vehicles])

veh = vehicles[0]
print("\n",
    f"Pos ({veh.getPosition().x}; {veh.getPosition().y}), "
    f"Speed {veh.getSpeed()}, "
    f"Heading {veh.getHeading()*180/3.14}, "
    f"Has collided {veh.getCollided()}, "
    f"Goal pos ({veh.getGoalPosition().x}; {veh.getGoalPosition().y})"
)

for _ in range(100):
    veh.setAccel(20.0)
    veh.setSteeringAngle(-12. * 3.1415 / 180.0)
    sim.step(0.1)

print("\n",
    f"Pos ({veh.getPosition().x}; {veh.getPosition().y}), "
    f"Speed {veh.getSpeed()}, "
    f"Heading {veh.getHeading()*180/3.14}, "
    f"Has collided {veh.getCollided()}, "
    f"Goal pos ({veh.getGoalPosition().x}; {veh.getGoalPosition().y})"
)