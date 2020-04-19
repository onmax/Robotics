import cv2
import numpy as np


class SceneMoments():
    def __init__(self, sections_img, color):
        self.contours = self.get_contours(sections_img, color)

        # Getting defects
        if len(self.contours) > 0:
            self.defects = self.get_defects()
        else: 
            self.defects = []
        
    def get_contours(self, sections_img, color):
        sections_img_bw = np.all(sections_img == color, axis=-1).astype(np.uint8)[90:,:] * 255
        contours, _ = cv2.findContours(sections_img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        return contours

    def get_defects(self):
        chull_list = [cv2.convexHull(contour,returnPoints=False) for contour in self.contours]
        defects = [cv2.convexityDefects(contour, chull) for (contour,chull) in zip(self.contours, chull_list)]


        if len(defects) == 0 or np.all(np.equal(defects[0], None)):
            return []

        contour = self.contours[0]
        defects = defects[0]
        
        defects = defects[:,0,:].tolist()
        defects = [[start, end, mid, length] for start, end, mid, length in defects if length > 1000]
        return defects