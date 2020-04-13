'''
AUTHORS

Máximo José García Martínez
Andrea Velarde Chávez
Alejandro Senovilla Tejedor
'''

from pyrobot.brain import Brain

import math



class BrainTestNavigator(Brain):
    NO_FORWARD = 0
    SLOW_FORWARD = 0.45
    MED_FORWARD = 0.75
    FULL_FORWARD = 1.5

    NO_TURN = 0
    MED_LEFT = 0.5
    HARD_LEFT = 1.0
    MED_RIGHT = -0.5
    HARD_RIGHT = -1.0

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

    # As in the slides, this variables are not used sadly   
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

    INTIAL_SPIRAL_SIZE = 4
    SPIRAL_SIZE = INTIAL_SPIRAL_SIZE
    TICKS_IN_SPIRAL = 0
    N_TURNS_IN_SPIRAL = 0

    LOCATIONS = ["left", "left-top", "top-left", "ttop-left", "ttop-right", "top-right", "right-top", "right"]
    SONARS = {}
    SONAR = {
        "ACTIVE": False,
        "WALL_IN": "",
        "HAS_BEEN_ACTIVE_FOR": 0, # Number of ticks that the SONAR has been active
        "LOOKING_FOR_LINE_FOR": 0 # Number of ticks that the robot has been looking for the line rotating
    }

    def get_active_sonars(self):
        # This function returns the sonars that have detected an object closer than 0.6.
        # It will return the values as a dictionary. As an example: {"left": 0.3, "left-top": 0.46}
        sonars_list = [x.value for x in self.robot.sonar[0]['all']]

        # Convert it to dictionary
        for (sonar , location) in zip(sonars_list, self.LOCATIONS):
            self.SONARS[location] = sonar
        
        # Filter then sonar to get only the ones that are close to the object
        return list(filter(lambda x: self.SONARS[x] < 0.6, self.SONARS))
    
    def rotate_to_go_around(self, active_sonars):
        # This function will rotate around an object.
        # It will produce 3 moves in the robot depending on the sensors:
        # - 1. It will go straigt if it detects that the object is on the left right
        # - 2. It will turn right in order to try to leave the object on the left
        # - 3. It will turn left in order to try to leave the object on the left

        self.SONAR["ACTIVE"] = True

        def is_in_active(l):
            return len(list(set(l)&set(active_sonars))) > 0

        print(active_sonars, is_in_active(['left', 'right']))
        if is_in_active(['left', 'right']) and len(active_sonars) <= 2:
            self.move(0.7, 0)    
        else:
            if is_in_active(["left-top", "top-left", "ttop-left"]):
                self.move(0, -0.2)
                self.SONAR["WALL_IN"] = "LEFT"
            else:
                self.move(0, 0.2)
                self.SONAR["WALL_IN"] = "RIGHT"


    def go_around(self):
        # This function will have two states:
        # 1. Search for the line rotating 90º to the opposite side where the object is placed and going back again if the line is not found. Example 1.
        # 2. Go around the object. If the right or left sensor stops reading data from the sensor
        # (it gets the maximum value), that means the robot needs to turn in order to keep around the object. Example 2

        # EXAMPLES
        # The line represent the sensor
        #  1.           2.
        #    ROBOT              ROBOT
        #     |                 |
        #   XXXXXX       XXXXX  |
        if self.SONAR["HAS_BEEN_ACTIVE_FOR"] % 2 == 0:
            self.SONAR["LOOKING_FOR_LINE_FOR"] += 1
            rotate = -5 if self.SONAR["WALL_IN"] == "RIGHT" else 5
            if self.SONAR["LOOKING_FOR_LINE_FOR"] == 1:
                self.move(0, rotate)
            else:
                self.move(0, -rotate)
                self.SONAR["LOOKING_FOR_LINE_FOR"] = 0
                self.SONAR["HAS_BEEN_ACTIVE_FOR"] += 1
        else:
            self.SONAR["HAS_BEEN_ACTIVE_FOR"] += 1

            if self.SONAR["WALL_IN"] == "RIGHT":
                print("Avoiding obstacle turning right")
                self.move(0, -3)
            elif self.SONAR["WALL_IN"] == "LEFT":
                print("Avoiding obstacle turning left")
                self.move(0, 3)
            else:
                self.move(1,0)

    def setup(self):
        pass
    

    def step(self):
        active_sonars = self.get_active_sonars()
        if len(active_sonars) > 0:
            self.rotate_to_go_around(active_sonars)
            return

        line_is_visible, error, searchRange = eval(
            self.robot.simulation[0].eval("self.getLineProperties()"))
        print("Line: {}. Error: {}".format(line_is_visible, error))
        
        if self.SONAR["ACTIVE"] and not line_is_visible:
            self.go_around()
            return

        self.SONAR["ACTIVE"] = False

        #  UNUSED VARIABLES
        derivative = error - self.PREVIOUS_ERRORS[-1]
        self.INTEGRAL += error
        self.PREVIOUS_ERRORS.append(error)

        if line_is_visible:
            self.N_TICKS_LOST = 0
            self.SEARCH_LEFT = 0
            self.SEARCH_RIGTH = 0
            self.SPIRAL_SIZE = self.INTIAL_SPIRAL_SIZE
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
            # Going in spiral if the search around the robot didn't work.
            # It will go forward for a certain amount of ticks. Then, it will turn and go forward for a certain amount of ticks again
            # Every 4 of this iterations it will increase the number of TICKS that the robot needs to go forward.
            self.TICKS_IN_SPIRAL += 1
            print("Spiral size: {}. Ticks in spiral: {}".format(self.SPIRAL_SIZE, self.TICKS_IN_SPIRAL))
            if self.TICKS_IN_SPIRAL % self.SPIRAL_SIZE == 0:
                self.move(0, 5)
                self.N_TURNS_IN_SPIRAL += 1
                if self.N_TURNS_IN_SPIRAL % 4 == 0:
                    self.SPIRAL_SIZE += int(self.INTIAL_SPIRAL_SIZE * 1.5)
                return
            self.move(1, 0)
        else:
            # It will search for the left and the for the right.
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
