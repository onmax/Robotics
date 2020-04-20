'''
- Command control for the line.
    1. if number of contours in complementary is 0 or 1 => Nothing
    2. if number of contours in complementary is 2 => Straight line or curve
        2.1 if number of defects is 0 => Straight line
        2.2 else => curve
            2.2.1 If the sum of boundaries is 2, and at least one of them is from the bottom and the other one is going to on of the sides. If the side is left => curve left, or if it is right => curve right
            2.2.2 Fit a ellipsis in the contour and get the value of inclination. If angle is negative => curve left. If angle is positive => curve right
    3. If number of contours in complementary is 3 => two-way street
    4. If number of contours in complementary is 4 or more => three-way street. There is no four-way street or more.
'''

import cv2 
import numpy as np


class Path():
    path_str = ""

    def set_params(self, is_straight_line=False, curve_direction=None, n_way_street=None):
        self.is_straight_line = is_straight_line
        self.curve_direction = curve_direction
        self.n_way_street = n_way_street

    def nothing(self):
        self.path_str = "Nothing"
        self.set_params()
        return self
    
    def set_is_straight_line(self):
        self.path_str = "Straight line"
        self.set_params(is_straight_line=True)
        return self

    def set_curve(self, direction):
        self.path_str = "{} curve".format(direction)
        self.set_params(curve_direction=direction)
        return self
        
    def set_n_way_cross(self, n_way_street):
        self.path_str = "{}-way-street".format(n_way_street)
        self.set_params(n_way_street=n_way_street)
        return self
    
    def __str__(self):
        return self.path_str

class ControlCommand():

    def __init__(self, sections_img, boundaries, sm_line, sm_sign):
        self.path = self.detect_path(sections_img, boundaries, sm_line)

    def detect_path(self, sections_img, boundaries, sm_line):
        # step 1
        if len(sm_line.contours_compl) <= 1:
            return Path().nothing()
        
        # step 2
        if len(sm_line.contours_compl) == 2:
            if len(sm_line.defects) == 0:
                return Path().set_is_straight_line()
            else:
                if boundaries.bottom == 1 and boundaries.left == 1:
                    return Path().set_curve("Left")
                elif boundaries.bottom == 1 and boundaries.right == 1:
                    return Path().set_curve("Right")
                else:
                    return Path().set_is_straight_line()

    
        # step 3 and 4
        if len(sm_line.contours_compl) > 2:
            return Path().set_n_way_cross(min(len(sm_line.contours_compl), 4))
        
        return Path().nothing()
    
    def sstr(self):
        return [str(self.path)]







