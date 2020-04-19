import cv2
import numpy as np


class SceneMoments():
    def __init__(self, sections_img, color, min_contour_size=1000, type_object=""):
        self.min_contour_size = min_contour_size

        self.type_object = type_object
        self.contours = self.get_contours(sections_img, color)

        # Getting defects
        if len(self.contours) > 0:
            self.defects = self.get_defects()
        else: 
            self.defects = []
        
    def get_contours(self, sections_img, color):
        sections_img_bw = np.all(sections_img == color, axis=-1).astype(np.uint8)[90:,:] * 255
        a = np.all(sections_img == color, axis=-1).astype(np.uint8)[90:,:] 
        sections_img_bw_compl = a^(a&1==a) * 255
        cv2.imshow("Images", sections_img_bw_compl)
        cv2.waitKey(0)

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
        defects = [[start, end, mid, length] for start, end, mid, length in defects if length > self.min_contour_size]
        return defects
    
    def __str__(self): 
        lengths = "{} contours: {}  defects: {}".format(self.type_object, len(self.contours), len(self.defects))
        return lengths
    
    def paint_defects(self, img, color):
        if len(self.contours) == 0:
            return img
        contour = self.contours[0]
        for s, e, m, l in self.defects:
            cv2.circle(img, (contour[m][0][0], contour[m][0][1] + 90), 5, color, -1)
        return img

    def paint_lines(self, img, color):
        if len(self.contours) == 0:
            return img
        contour = self.contours[0]
        for s, e, m, l in self.defects:
            cv2.line(img, (contour[s][0][0], contour[s][0][1] + 90), (contour[e][0][0], contour[e][0][1] + 90), color, 2)
        return img