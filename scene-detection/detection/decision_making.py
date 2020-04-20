class DecisionMaking():
    def __init__(self, memory):
        last_state = memory[-1]
        if last_state.path.n_way_street != None:
            if last_state.signs.arrow != None:
                self.decision = "Take the {} exit".format(last_state.signs.arrow.direction)
            else:
                # Check for the last 30 frames and get the most recent with an arrow
                previous_state_with_arrow = [s for s in reversed(memory[1:30]) if s.signs.arrow != None]
                if len(previous_state_with_arrow) != 0:
                    state = previous_state_with_arrow[0]
                    self.decision = "Take the {} exit".format(state.signs.arrow.direction)
                else:
                    self.decision = "Go straight"
        else:
            if last_state.path.is_straight_line:
                self.decision = "Go straight"
            elif last_state.path.curve_direction != None:
                self.decision = "Turn {}".format(last_state.path.curve_direction)
            else:
                self.decision = "Go straight"
        
