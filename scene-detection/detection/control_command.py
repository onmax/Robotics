import numpy as np
import cv2

class ControlCommand():
    def __init__(self, memory):
        # self.memory = memory

        self.current_state = memory[-1]

        self.angle = self.get_angle(self.current_state, memory)
        print(self.angle)
        self.vx = abs(np.sin(self.angle * np.pi / 180))
        self.vy = abs(np.cos(self.angle * np.pi / 180))


    '''
    It returns the angle of the most recent arrow detected in the last 100 frames. If no arrow is detected, an angle of 0 will be returned
    '''
    def arrow_angle(self, state, memory):
        previous_state_with_arrow = [s for s in reversed(memory[-101:-1]) if s.signs.arrow != None]
        if len(previous_state_with_arrow) != 0:
            state_with_arrow = previous_state_with_arrow[0]
            return state_with_arrow.signs.arrow.angle
        else:
            return 0

    def get_angle_between_2_boundaries(self, entrance_boundary, destination_boundary):
        x1, y1 = entrance_boundary.mid[0], entrance_boundary.mid[1]
        x2, y2 = destination_boundary.mid[0], destination_boundary.mid[1]
        print(x1, y1, x2, y2)

        vector_dir = np.array([x1-x2 , y1-y2])
        norm_dir = np.linalg.norm(vector_dir)
        vector_vertical = np.array([0,1])
        norm_vertical = np.linalg.norm(vector_vertical)
        
        angle = np.degrees(np.arccos((np.dot(vector_dir,vector_vertical)) / (norm_dir * norm_vertical)))
        angle = angle * -1 if x2 < x1 else angle

        return angle
    
    def get_closest_boundary_to_angle(self, active, small_boundaries, angle):
        candidates = small_boundaries.get_boundaries_no_bottom()
        if active == None:
            return None
        angles_right = [self.get_angle_between_2_boundaries(active, b) for b in small_boundaries.right]
        angles_left = [self.get_angle_between_2_boundaries(active, b) for b in small_boundaries.left]
        angles_top = [self.get_angle_between_2_boundaries(active, b) for b in small_boundaries.top]
        
        min_angle = 90
        for _angle in angles_right + angles_top + angles_left:
            if abs(angle - abs(_angle)) < abs(min_angle):
                min_angle = _angle
        return min_angle

    '''
    It will set the destination points. For that it will use the boundaries of the small square. If the small square contains 2 boundaries, then the boundary that is not in the bottom will be the destination (we assume that always there is one boundary at the bottom)
    '''
    def get_angle(self, state, memory):
        small_boundaries = state.description.small_boundaries
        if state.path.n_way_street != None and small_boundaries.total > 2:
            # It is a n-street
            angle_o = self.arrow_angle(state, memory)
        elif small_boundaries.total == 2:
            # straight or curve
            angle_o = 0
        
        angle = self.get_closest_boundary_to_angle(state.description.boundaries.get_active_lane(), small_boundaries, angle_o)
        return angle
    
    def paint_vector(self, image):
        b = self.current_state.description.boundaries.get_active_lane()
        x1, y1 = b.mid[0], b.mid[1]

        x2 = int(x1 - 60 * -np.sin(self.angle * np.pi / 180))
        y2 = int(y1 + 60 * -np.cos(self.angle * np.pi / 180))
        image = cv2.line(image, (x1, y1), (x2, y2), (120, 0, 120), 2)
        return image

    def __str__(self):
        return "VX: {}   VY:{}".format(self.vx, self.vy)