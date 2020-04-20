import cv2
import numpy as np


class SceneMoments():
    def __init__(self, sections_img, color, min_contour_size=1000, type_object=""):
        self.min_contour_size = min_contour_size

        self.type_object = type_object

        bw = np.all(sections_img == color, axis=-1).astype(np.uint8)[90:,:]

        sections_bw = bw * 255
        self.contours, self.defects = self.contours_n_defects(sections_bw)

        sections_bw_compl = bw - 1
        self.contours_compl, self.defects_compl = self.contours_n_defects(sections_bw_compl)

    
    def contours_n_defects(self, img_bw):
        contours, _ = cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            return contours, []

        chull_list = [cv2.convexHull(contour,returnPoints=False) for contour in contours]
        defects = [cv2.convexityDefects(contour, chull) for (contour,chull) in zip(contours, chull_list)]

        if len(defects) == 0 or np.all(np.equal(defects[0], None)):
            return contours, []

        contour = contours[0]
        defects = defects[0]
        
        defects = defects[:,0,:].tolist()
        defects = [[start, end, mid, length] for start, end, mid, length in defects if length > self.min_contour_size]
        return contours, defects
    
    '''
    Return a list of string containing useful data about the object: length of the contours and length of the defects for both, complement and normal
    '''
    def sstr(self): 
        lengths = "{} contours: {}  defects: {}".format(self.type_object, len(self.contours), len(self.defects))
        lengths_compl = "Compl contours: {}  defects: {}".format(len(self.contours_compl), len(self.defects_compl))
        return [lengths, lengths_compl, ""]
    
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
            print("lol", contour[s][0])
            cv2.line(img, (contour[s][0][0], contour[s][0][1] + 90), (contour[e][0][0], contour[e][0][1] + 90), color, 2)
        return img