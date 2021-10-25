import os

import matplotlib.pyplot as plt
import numpy as np
from nocturne import Simulation

import time

os.environ["DISPLAY"] = ":0.0"

sim = Simulation(scenarioPath='./scenarios/intersection.json')
scenario = sim.getScenario()

dt = 1.0 / 30.0
for i in range(50):
    sim.step(dt)
    sim.render()
    time.sleep(dt)

sim.saveScreenshot()