import cv2
import numpy as np
from detection.boundaries import Boundaries
from detection.scene_moments import SceneMoments
from detection.scene_state import SceneState
from detection.control_command import ControlCommand

class SceneDescription():
    def __init__(self, image, memory, w=60, h=60):
        self.image = image
        self.boundaries = Boundaries(self.image)
        self.scene_moments_line = SceneMoments(self.image, [255, 0, 0], type_object="line", compl=True)
        self.scene_moments_signs = SceneMoments(image, [0, 0, 255], min_contour_size=1000, type_object="sign")
        self.set_active_lane(memory)

        self.small_image = self.get_small_image(w, h)
        if len(self.boundaries.bottom) > 0:
            self.small_boundaries = self.set_small_boundaries()
        else:
            self.small_boundaries = self.set_small_boundaries()

    def paint_verbose(self, image):
        image = self.scene_moments_line.paint_contours(image, [255, 150, 36])
        # image = self.scene_moments_line.paint_lines(image, [255, 255, 0])
        # image = self.scene_moments_line.paint_defects(image, [255, 0, 255])
        # image = self.scene_moments_signs.paint_lines(image, [0, 255, 255])
        # image = self.scene_moments_signs.paint_defects(image, [0, 120, 255])
        image = self.paint_small_square(image)
        image = self.small_boundaries.paint_boundaries_mid(image)
        image = self.boundaries.paint_boundaries_mid(image)
        return image
        
    def set_active_lane(self, memory):
        if len(self.boundaries.bottom) == 1:
            self.boundaries.bottom[0].current_lane = True
        elif len(self.boundaries.bottom) >= 2:
            self.active_lane_from_memory(memory)

    def active_lane_from_memory(self, memory):
        m100 = [m.description.boundaries.get_active_lane() for m in memory[-60:-10]]
        mean = np.mean(np.array([b.mid[0] for b in m100 if b]))
        closest = np.abs([b.mid[0] for b in self.boundaries.bottom] - mean).argmin()
        self.boundaries.bottom[closest].current_lane = True
    
    def get_small_image(self, w, h):
        if len(self.boundaries.bottom) == 0:
            return None
        mid = [b.mid[0] for b in self.boundaries.bottom if b.current_lane][0]
        self.w, self.h = w, h
        self.w1, self.w2 = max(int(mid - w / 2), 0), min(int(mid + w / 2), 359)
        self.h1, self.h2 = int(self.image.shape[0]) - h, int(self.image.shape[0])
        return self.image[self.h1:self.h2, self.w1:self.w2,:]

    def set_small_boundaries(self):
        small_boundaries = Boundaries(self.small_image)
        small_boundaries.apply_offset(self.h2 - self.w, self.w1)
        return small_boundaries

    def paint_small_square(self, image):
        sp = (self.w1, self.h1)
        ep = (self.w2, self.h2 + self.h)
        image = cv2.rectangle(image, sp, ep, [25, 255, 251], 1)
        return image

    def sstr(self):
        return [str(self.boundaries), ""] + self.scene_moments_line.sstr() + self.scene_moments_signs.sstr()
