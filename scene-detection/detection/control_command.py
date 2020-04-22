import numpy as np

class ControlCommand():
    def __init__(self, memory):
        self.memory = memory
        self.memory[-1].boundaries.bottom = self.set_active_lane()

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
        

    def set_active_lane(self):
        bottom = self.memory[-1].boundaries.bottom
        if len(bottom) == 1:
            bottom[0].current_lane = True
        elif len(bottom) >= 2:
            bottom = self.active_lane_from_memory(bottom)
        return bottom

    def active_lane_from_memory(self, bottom):
        m100 = [m.boundaries.get_active_lane() for m in self.memory[-110:-10]]
        mean = np.mean(np.array([b.mid[0] for b in m100 if b]))
        closest = np.abs([b.mid[0] for b in bottom] - mean).argmin()
        bottom[closest].current_lane = True
        return bottom
