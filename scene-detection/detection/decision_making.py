class DecisionMaking():
    def __init__(self, control_command):
        last_state = control_command[-1]
        if last_state.signs.arrow != None:
            self.decision = "Take the {} exit".format(last_state.signs.arrow.direction)
        else:
            if last_state.path.is_straight_line:
                self.decision = "Go straight"
            elif last_state.path.curve_direction != None:
                self.decision = "Turn {}".format(last_state.path.curve_direction)
            else:
                self.decision = "Go straight"
        
