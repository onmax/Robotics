from pyrobot.brain import Brain

import math
import numpy as np

class BrainTestNavigator(Brain):
    NO_FORWARD = 0
    SLOW_FORWARD = 0.45
    MED_FORWARD = 0.75
    FULL_FORWARD = 1.5

    NO_TURN = 0

    SLOW_LEFT = 0.1
    MED_LEFT = 0.45
    HARD_LEFT = 0.8

    SLOW_RIGHT = -0.1
    MED_RIGHT = -0.45
    HARD_RIGHT = -0.8

    STATES = [
        { "min": 1, "max": 0.6, "forward": NO_FORWARD, "direction": HARD_LEFT},
        { "min": 0.6, "max": 0.3, "forward": SLOW_FORWARD, "direction": MED_LEFT},
        { "min": 0.3, "max": 0.05, "forward": MED_FORWARD, "direction": SLOW_LEFT},
        { "min": 0.05, "max": -0.05, "forward": FULL_FORWARD, "direction": NO_TURN},
        { "min": -0.05, "max": -0.3, "forward": MED_FORWARD, "direction": SLOW_RIGHT},
        { "min": -0.3, "max": -0.6, "forward": SLOW_FORWARD, "direction": MED_RIGHT},
        { "min": -0.6, "max": -1, "forward": NO_FORWARD, "direction": HARD_RIGHT},
    ]

    # As in the slides
    PREVIOUS_ERRORS = [0] * 10
    INTEGRAL = 0

    # This variable stores the state that was used in the previous tick 
    LAST_STATE = STATES[3]

    # This is how many ticks the robot has been lost
    # Specifically, the number of ticks that have been missing without finding the line
    N_TICKS_LOST = 0

    TICKS_TO_TURN_180 = int((math.pi**2))
    SEARCH_LEFT = 0
    SEARCH_RIGTH = 0

    SPIRAL_SIZE = 4
    TICKS_IN_SPIRAL = 0
    N_TURNS_IN_SPIRAL = 0

    def setup(self):
        pass

    def step(self):
        line_is_visible, error, searchRange = eval(
            self.robot.simulation[0].eval("self.getLineProperties()"))
        print("Line: {}. Error: {}".format(line_is_visible, error))

        # We might use this variables
        derivative = error - self.PREVIOUS_ERRORS[-1]
        self.INTEGRAL += error
        self.PREVIOUS_ERRORS.append(error)

        if line_is_visible:
            self.N_TICKS_LOST = 0
            self.SEARCH_LEFT = 0
            self.SEARCH_RIGTH = 0
            self.SPIRAL_SIZE = 4
            self.N_TURNS_IN_SPIRAL = 0
            self.TICKS_IN_SPIRAL = 0
            
            # This part calculates the amount of throttle and the amount of steering that 
            # needs to be applied in order to minimaze the error (go to the center of the
            # line)
            normalized_error = error / searchRange
            for state in self.STATES:
                if state["min"] >= normalized_error > state["max"]:
                    self.move(state["forward"], state["direction"])
                    self.LAST_STATE = state
                    print("Turning {}: {}".format("right" if state["direction"] < 0 else "left",  state["direction"]))
        elif self.SEARCH_LEFT >= self.TICKS_TO_TURN_180 and self.SEARCH_RIGTH  >= self.TICKS_TO_TURN_180 * 3:
            self.TICKS_IN_SPIRAL += 1

            print("Spiral size: {}. Ticks in spiral: {}".format(self.SPIRAL_SIZE, self.TICKS_IN_SPIRAL))

            if self.TICKS_IN_SPIRAL % self.SPIRAL_SIZE == 0:
                self.move(0, 5)
                self.N_TURNS_IN_SPIRAL += 1
                if self.N_TURNS_IN_SPIRAL % 4 == 0:
                    self.SPIRAL_SIZE += 4
                return

            self.move(1, 0)
        else:
            if self.SEARCH_LEFT <= self.TICKS_TO_TURN_180:
                print("Lost. Searching left...")
                self.move(self.NO_FORWARD, 1)
                self.SEARCH_LEFT += 1
            else:
                print("Lost. Searching right...")
                self.move(self.NO_FORWARD, -1)
                self.SEARCH_RIGTH += 1

            self.N_TICKS_LOST += 1   


def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))

    # If we are allowed (for example you can't in a simulation), enable
    # the motors.
    try:
        engine.robot.position[0]._dev.enable(1)
    except AttributeError:
        pass

    return BrainTestNavigator('BrainTestNavigator', engine)
