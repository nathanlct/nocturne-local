import numpy as np
from nocturne import Simulation
import matplotlib.pyplot as plt
import time

sim = Simulation(scenarioPath='./scenarios/basic.json')
scenario = sim.getScenario()

dt = 1.0 / 30.0
while True:
    sim.step(dt)
    sim.render()
    time.sleep(dt)
