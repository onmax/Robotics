from pyrobot.brain import Brain

import math
import numpy as np

class BrainTestNavigator(Brain):
    NO_FORWARD = 0.05
    SLOW_FORWARD = 0.35
    MED_FORWARD = 0.6
    FULL_FORWARD = 1

    NO_TURN = 0

    NO_LEFT = 0
    SLOW_LEFT = 0.1
    MED_LEFT = 0.45
    HARD_LEFT = 0.8

    NO_RIGHT = 0
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
    PREVIOUS_ERROR = 0
    INTEGRAL = 0

    # This variable stores the state that was used in the previous tick 
    LAST_STATE = STATES[3]

    # This is how many ticks the robot has been lost
    # Specifically, the number of ticks that have been missing without finding the line
    N_TICKS_LOST = 0

    def setup(self):
        pass

    def step(self):
        line_is_visible, error, searchRange = eval(
            self.robot.simulation[0].eval("self.getLineProperties()"))
        print("Line: {}. Error: {}. Search range: {}".format(line_is_visible, error, searchRange))

        # We might use this variables
        derivative = error - self.PREVIOUS_ERROR
        self.INTEGRAL += error

        if line_is_visible:
            self.N_TICKS_LOST = 0

            # This part calculates the amount of throttle and the amount of steering that 
            # needs to be applied in order to minimaze the error (go to the center of the
            # line)
            normalized_error = error / searchRange
            for state in self.STATES:
                if state["min"] >= normalized_error > state["max"]:
                    self.move(state["forward"], state["direction"])
                    self.LAST_STATE = state
        else:
            # If the line is not found, the robot won't move, it is going to steer
            # to the direction where the line was seen last time
            if self.N_TICKS_LOST < 14:
                steering = 0.4
                direction = steering if self.LAST_STATE["direction"] < 0 else -steering
                self.move(0,direction)
                self.N_TICKS_LOST += 1

        self.PREVIOUS_ERROR = error

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
