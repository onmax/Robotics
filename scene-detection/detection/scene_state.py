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
from functions import top_offset
from shape_detection.train_shape import KneighboursClassifier

class Path():
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

    def set_n_way_street(self, n_way_street):
        self.path_str = "{}-way-street".format(n_way_street)
        self.set_params(n_way_street=n_way_street)
        return self

    def __str__(self):
        return self.path_str


class Arrow():
    def __init__(self, center_ellipsis, center_masses, angle_ellipsis):
        (EX, EY) = center_ellipsis
        (MX, MY) = center_masses
        self.angle = -(angle_ellipsis - 90) if EX > MX else angle_ellipsis

        self.direction = "left" if self.angle < 0 else "right"


class Signs():
    def __init__(self):
        self.arrow = None

    def set_arrow(self, center_ellipsis, center_masses, ellipsis_angle):
        self.arrow = Arrow(center_ellipsis, center_masses, ellipsis_angle)
        self.sign_str = "Arrow {} degrees pointing to {}".format(
            round(self.arrow.angle, 2), self.arrow.direction)
        return self

    def nothing(self):
        self.sign_str = "No signs"
        return self

    def normal_sign(self, image, model):
        prediction = model.predict(image)
        if len(prediction) == 0:
            self.sign_str = "Unknown sign"
        else:
            self.sign_str = prediction[0]
        return self

    def __str__(self):
        return self.sign_str


class SceneState():
    def __init__(self, scene_description, frame_n, model, debug_mode=False):
        self.debug_mode = debug_mode
        self.frame_n = frame_n

        image = scene_description.image
        boundaries = scene_description.boundaries
        sm_line = scene_description.scene_moments_line
        sm_sign = scene_description.scene_moments_signs
        self.description = scene_description
        self.path = self.detect_path(boundaries, sm_line)
        self.signs = self.detect_signs(scene_description.image, sm_sign, model)

    def detect_path(self, boundaries, sm_line):
        # step 1
        if len(sm_line.contours_compl) <= 1:
            return Path().nothing()

        # step 2
        if len(sm_line.contours_compl) == 2:
            if len(sm_line.defects) == 0:
                return Path().set_is_straight_line()
            else:
                if len(boundaries.bottom) == 1 and len(boundaries.left) == 1:
                    return Path().set_curve("Left")
                elif len(boundaries.bottom) == 1 and len(boundaries.right) == 1:
                    return Path().set_curve("Right")
                else:
                    return Path().set_is_straight_line()

        # step 3 and 4
        if len(sm_line.contours_compl) > 2:
            return Path().set_n_way_street(min(len(sm_line.contours_compl), 4))

        return Path().nothing()

    def detect_arrow(self, image, sm_sign):
        contour = sm_sign.contour
        if len(sm_sign.contours) != 1 or len(contour) < 5:
            return Signs().nothing()

        ellipse = cv2.fitEllipse(contour)
        cE = (int(ellipse[0][0]), int(ellipse[0][1]) + top_offset)
        ellipsis_angle = ellipse[2]

        M = cv2.moments(contour)
        cX = int(M["m10"] / (M["m00"] + 1e-5))
        cY = int(M["m01"] / (M["m00"] + 1e-5)) + top_offset

        if self.debug_mode:
            cv2.circle(image, (cX, cY), 2, (255, 255, 255), -1)  # white masses
            cv2.circle(image, cE, 2, (0, 0, 0), -1)
        return Signs().set_arrow(cE, (cX, cY), ellipsis_angle)

    def detect_signs(self, image, sm_sign, model):
        if len(sm_sign.contours) == 0:
            return Signs().nothing()
        if self.path.n_way_street != None:
            return self.detect_arrow(image, sm_sign)
        else:
            return Signs().normal_sign(image, model)

    def sstr(self):
        return [str(self.path), str(self.signs)]
