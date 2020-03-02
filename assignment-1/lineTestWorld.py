"""
A simulation with a simple line on the floor.
"""

from pyrobot.simulators.lineSimulation import LineSimulation
from pyrobot.simulators.pysim import TkPioneer, \
     PioneerFrontSonars, PioneerFrontLightSensors



def INIT():
    background = 3
    positions = [[1, 18.9, 4.0], [8.5, 2.35, 1.57], [10.5, 19, 8.2], [0.43, 10.7, 12]]

    # (width, height), (offset x, offset y), scale
    sim = LineSimulation((450, 675), (20, 650), 32,
                         background="./lineBackground{}.png".format(background))

    # an example of an obstacle on the line
    # x1, y1, x2, y2
    sim.addBox(5, 12, 6, 11)
    sim.addBox(5, 9, 5.4, 9.4)
    
    sim.addRobot(60000,
                 # name, x, y, th, boundingBox
                 TkPioneer("RedErratic",
                           positions[background-1][0], 
                           positions[background-1][1],
                           positions[background-1][2],
                           ((.185, .185, -.185, -.185),
                            (.2, -.2, -.2, .2))))

    # add some sensors:
    sim.robots[0].addDevice(PioneerFrontSonars())

    # to create a trail
    sim.robots[0].display["trail"] = 1

    return sim
