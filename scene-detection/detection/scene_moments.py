import cv2
import numpy as np
from functions import top_offset

class SceneMoments():
    def __init__(self, sections_img, color, min_contour_size=1000, type_object="", offset=True, compl=False):
        self.min_contour_size = min_contour_size

        self.type_object = type_object

        self.bw = np.all(sections_img == color, axis=-1).astype(np.uint8)
        if offset:
            self.bw = self.bw[top_offset:,:]
        cv2.waitKey(0)

        sections_bw = self.bw * 255
        self.contours = self.get_contours(sections_bw)
        self.contour, self.contour_index = self.get_contour()
        self.defects = self.get_defects()

        self.compl = compl
        if compl:
            sections_bw_compl = 255 - sections_bw
            self.contours_compl = self.get_contours(sections_bw_compl)
            self.defects_compl = self.get_defects()

    
    def get_contours(self, img_bw):
        contours, _ = cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        return contours

    def get_defects(self):
        chull_list = [cv2.convexHull(contour,returnPoints=False) for contour in self.contours]
        defects = [cv2.convexityDefects(contour, chull) for (contour,chull) in zip(self.contours, chull_list)]

        if len(defects) == 0 or np.all(np.equal(defects[0], None)):
            return []

        defect = defects[self.contour_index]
        if not isinstance(defect, np.ndarray):
            return []

        defect = defect[:,0,:].tolist()
        defect = [[start, end, mid, length] for start, end, mid, length in defect if length > self.min_contour_size]
        return defect
    
    '''
    Return the largest contour
    '''
    def get_contour(self):
        if len(self.contours) == 0:
            return [], -1
        largest_area = 0
        contour = self.contours[0]
        i = 0
        for _i, _contour in enumerate(self.contours):
            area = cv2.contourArea(_contour) 
            if area > largest_area:
                largest_area = area
                contour = _contour
                i = _i
        return contour, i
        # return self.contours[0], 0

    '''
    Return a list of string containing useful data about the object: length of the contours and length of the defects for both, complement and normal
    '''
    def sstr(self):

        if not self.compl:
            lengths = "{} contours: {}  defects: {}".format(self.type_object, len(self.contours), len(self.defects))
            return [lengths, ""]
        else:
            lengths = "{} contours: {}  defects: {}".format(self.type_object, len(self.contours), len(self.defects))
            lengths_compl = "Compl contours: {}  defects: {}".format(len(self.contours_compl), len(self.defects_compl))
            return [lengths, lengths_compl, ""]
    
    def paint_defects(self, img, color):
        if len(self.contours) == 0:
            return img
        contour = self.contour
        for s, e, m, l in self.defects:
            cv2.circle(img, (contour[m][0][0], contour[m][0][1] + top_offset), 5, color, -1)
        return img

    def paint_lines(self, img, color):
        if len(self.contours) == 0:
            return img
        contour = self.contour
        for s, e, m, l in self.defects:
            cv2.line(img, (contour[s][0][0], contour[s][0][1] + top_offset), (contour[e][0][0], contour[e][0][1] + top_offset), color, 2)
        return img
    
    def paint_contours(self, img, color):
        if len(self.contours) == 0:
            return img
        contour = self.contour
        contour = np.array([[[c[0][0], c[0][1] + top_offset]] for c in contour])
        cv2.drawContours(img, contour, -1, color, 2)
        return img