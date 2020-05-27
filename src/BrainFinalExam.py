from pyrobot.brain import Brain

import math
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

import glob
import numpy as np
# import imageio


b, g, r = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
top_offset = 30

def open_images(images_paths):
    '''
    Returns a list of tuples containing (original_image, image_labelled)
    '''
    data = []

    for (path_original, path_train) in images_paths:
        print("Reading images:", path_train, path_original)

        original = np.array(cv2.imread(path_original))
        label = np.array(cv2.imread(path_train))
        data.append((original, label))

    return data


def predicted_in(start, end, n_frames):
    n_seconds = end - start
    print("Predicted in {} seconds {} frames. That is {} seconds/frame".format(
        n_seconds, n_frames, n_seconds / n_frames))

def write_text(img, texts):
    # https://gist.github.com/aplz/fd34707deffb208f367808aade7e5d5c
    bck = (0, 0, 0)
    color = (255, 255, 255)
    
    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_scale = 0.7
    texts_sizes = [cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0] for text in texts]

    text_width, text_height = max([s[0] for s in texts_sizes]), sum([s[1] for s in texts_sizes]) + 10 * (len(texts) - 1)

    padding = 6
    box_coords = ((0,0), (text_width + padding, text_height + padding + 15))

    img = cv2.rectangle(img, box_coords[0], box_coords[1], bck, cv2.FILLED)
    
    for i, text in enumerate(texts):
        padding_top = 0 if i == 0 else sum(s[1] for s in texts_sizes[:i]) + 10 * i
        img = cv2.putText(img, text, (padding, padding + 15 + padding_top), font, fontScale=font_scale, color=color, thickness=1)

    return img
  
class TrainShape():
    def get_img_n_labels(self):
        img_names = sorted(glob.glob('./images/shape/*.png'))
        print(np.unique(img_names))
        img_raw_list = np.array([cv2.imread(image) for image in img_names])
        labels = np.array([x.split("/")[-1].split("-")[0] for x in img_names])
        return img_raw_list, labels

    def k_fold(self):
        X, y = self.get_img_n_labels()
        loo = LeaveOneOut()
        predictions = []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index][0]
            y_train, y_test = y[train_index], y[test_index][0]
            clf = KneighboursClassifier().fit(X_train, y_train)
            predictions.append(clf.predict(X_test) == y_test)
            print(clf.predict(X_test), y_test)
        print(np.unique(predictions, return_counts=True))

    def train(self):
        img_raw_list, labels = self.get_img_n_labels()
        return KneighboursClassifier().fit(img_raw_list, labels)

class KneighboursClassifier():
    def fit(self, img_raw_list, labels):
        X = []
        for img_raw in img_raw_list:
            X.append(self.contour2des(img_raw, self.img2contour(img_raw)))
        X = np.array(X).reshape((-1, 32)).astype(np.uint8)
        self.neigh = KNeighborsClassifier(
            n_neighbors=5, metric=self.hamming_dist)
        # X son los datos de entrenamiento y son los datos objetivo
        self.neigh.fit(X, labels)
        return self

    def predict(self, img):
        X = self.contour2des(img, self.img2contour(img))
        if np.all(X == None):
            return []
        else:
            return self.neigh.predict(X)

    def hamming_dist(self, d1, d2):
        d1, d2 = d1.astype(np.uint8), d2.astype(np.uint8)
        assert d1.dtype == np.uint8 and d2.dtype == np.uint8
        d1_bits = np.unpackbits(d1)
        d2_bits = np.unpackbits(d2)
        return np.bitwise_xor(d1_bits, d2_bits).sum()

    def img2contour(self, img):
        img = img[:, :, :3]
        # bw = np.ones(img.shape[:2])
        # bw[np.where(np.all(img[:, :, 0] >= 230, img[:, :, 1]
        #    < 20, img[:, :, 2] < 20))] = 0
        bw = np.logical_and(img[:, :, 0] < 20, img[:, :, 1]
                            < 20, img[:, :, 2] >= 230).astype(np.uint8) * 255
        _,contours,_  = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        plottedContour = cv2.drawContours(bw,contours,-1,(0,255,0),2)
        cv2.imshow('CONTOUR',plottedContour)
        cv2.waitKey(0)
        contour = None
        if len(contours) != 0:
            for i in range(len(contours)):
                if len(contours[i]) >= 5:
                    contour = contours[i]
                    break

        return contour

    def contour2des(self, img, contour):
        (x, y), axis, angle = cv2.fitEllipse(contour)
        axis = np.array(axis)
        if angle > 90:
            angle -= 180
        orb = cv2.ORB_create()
        kp = cv2.KeyPoint(x, y, np.mean(axis) * 1.3, angle - 90)
        lkp, des = orb.compute(img, [kp])
        return des


class ControlCommand():
    def __init__(self, memory):
        # self.memory = memory

        self.current_state = memory[-1]
        # if self.current_state.signs.arrow != None:
        #     LAST_ARROW = self.current_state.signs.arrow

        self.angle = self.get_angle(self.current_state, memory)
        self.angle = 0 if self.angle == None else self.angle
        self.vx = np.sin(self.angle * np.pi / 180)
        self.vy = np.cos(self.angle * np.pi / 180)

    '''
    It returns the angle of the most recent arrow detected in the last 100 frames. If no arrow is detected, an angle of 0 will be returned
    '''

    def arrow_angle(self, state, memory):
        angles = [s.signs.arrow.angle for s in memory[-101:-20] if s.signs != None and s.signs.arrow != None]
        if len(angles) != 0:
            return np.mean(np.array(angles))
        else:
            return 0

    def get_angle_between_2_boundaries(self, entrance_boundary, destination_boundary):
        x1, y1 = entrance_boundary.mid[0], entrance_boundary.mid[1]
        x2, y2 = destination_boundary.mid[0], destination_boundary.mid[1]

        vector_dir = np.array([x1-x2, y1-y2])
        norm_dir = np.linalg.norm(vector_dir)
        vector_vertical = np.array([0, 1])
        norm_vertical = np.linalg.norm(vector_vertical)

        angle = np.degrees(
            np.arccos((np.dot(vector_dir, vector_vertical)) / (norm_dir * norm_vertical)))
        angle = angle * -1 if x2 < x1 else angle

        return angle

    def get_closest_boundary_to_angle(self, active, small_boundaries, angle):
        candidates = small_boundaries.get_boundaries_no_bottom()
        if active == None:
            return None
        angles_right = [self.get_angle_between_2_boundaries(
            active, b) for b in small_boundaries.right]
        angles_left = [self.get_angle_between_2_boundaries(
            active, b) for b in small_boundaries.left]
        angles_top = [self.get_angle_between_2_boundaries(
            active, b) for b in small_boundaries.top]

        if len(angles_right + angles_top + angles_left) == 0:
            return 0

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
        # elif small_boundaries.total == 2:
        else:
            # straight or curve
            angle_o = 0
        angle = self.get_closest_boundary_to_angle(
            state.description.boundaries.get_active_lane(), small_boundaries, angle_o)
        return angle

    def paint_vector(self, image):
        b = self.current_state.description.boundaries.get_active_lane()
        if b == None:
            return image
        x1, y1 = b.mid[0], b.mid[1]

        x2 = int(x1 - 60 * -np.sin(self.angle * np.pi / 180))
        y2 = int(y1 + 60 * -np.cos(self.angle * np.pi / 180))
        image = cv2.line(image, (x1, y1), (x2, y2), (120, 0, 120), 2)
        return image

    def sstr(self):
        return ["VX: {}".format(self.vx), "VY: {}".format(self.vy)]


class Boundary:
    def __init__(self, coords, x=None, y=None, current_lane=False):
        start, end, mid = coords
        if x == None:
            self.start = np.array([y, start])
            self.end = np.array([y, end])
            self.mid = np.array([y, mid])
        elif y == None:
            self.start = np.array([start, x])
            self.end = np.array([end, x])
            self.mid = np.array([mid, x])
        self.current_lane = current_lane

    def apply_offset(self, x, y):
        offset = np.array([y, x])
        self.start = self.start + offset
        self.end = self.end + offset
        self.mid = self.mid + offset

    def __str__(self):
        return "Start: {}, end: {}, mid: {}, current: {}".format(
            self.start, self.end, self.mid, self.current_lane
        )


class Boundaries:
    def __init__(self, sections_img):
        self.set_boundaries(sections_img)
        if len(self.bottom) == 0:
            width, height = sections_img.shape[:2]
            mid = int(width / 2)
            start, end = mid - 5, mid + 5
            self.bottom = [Boundary((start, end, mid), x=width)]

    '''
    This function wil read how many changes are in a array
    i.e. => [0, 255, 255, 255, 0, 255, 255, 0]
              ^              ^  ^         ^
    There are 4 changes => 2 paths. If the last and the first are differents is also a change.

    >>> v
    array([g,g,g,b,b,b,g,g,g,g,g,b,b,b,g,r,r,b,b,b])
    >>> start
    array([ 3, 11, 15])
    >>> end
    array([ 5, 13, 17])
    >>> mid
    array([ 4, 12, 16])
    '''

    def get_paths_boundaries(self, v):
        v[np.where(np.all(r == v, 1))] = g
        p = list(zip(v[:-1], v[1:]))
        start = [np.where(np.all(np.all(p == np.array([g, b]), axis=2), axis=1))[
            0] + 1][0]
        end = [
            np.where(np.all(np.all(p == np.array([b, g]), axis=2), axis=1))[0]][0]

        if np.all(v[0] == b):
            start = np.insert(start, 0, 0)

        if np.all(v[-1] == b):
            end = np.insert(end, len(end), len(v) - 1)

        mid = np.array([int(np.mean(x))
                        for x in np.array(list(zip(start, end)))])
        return list(zip(start, end, mid))

    '''
    If in a lane if it is in the corner exactly, get_paths_boundaries will detect two different lanes:
    one on each side of the corner. Let's say that the image is the following:

    [
        [g,g,g,g,b,b,b],
        [g,g,g,g,b,b,b],
        [g,g,b,b,b,g,g],
    ]

    b(blue) represents the lane. The top and right will have the following values:
    top = [<start: [0,4]; mid: []; end[0,6]; mid: [0,5]>]
    right = [<start: [0,6]; mid: []; end[1,6]; mid: [0,6]>]

    remove_one_in_the_corners will detect which lane is in the corner and the remove the vertical boundary and modify the boundary in the horizontal axis with the new values. the mid point will be the corner. With previous example, it will return:
    top = [<start: [0,4]; mid: []; end[1,6]; mid: [0,6]>]
    right = []

            beggining of the lane
    [            |
        [g,g,g,g,b,b,b],
        [g,g,g,g,b,b,b], -- end of the lane
        [g,g,b,b,b,g,g],
    ]
    '''

    def remove_one_in_the_corners(self, top, bottom, right, left):
        def equal(p1, p2):
            return p1[0] == p2[0] and p1[1] == p2[1]
        if len(bottom) > 0:
            # bottom right corner
            if len(right) > 0 and equal(bottom[-1].end, right[-1].end):
                bottom[-1].end = right[-1].start
                bottom[-1].mid = np.median(
                    np.array([bottom[-1].start, bottom[-1].end]), axis=0).astype(np.uint8)
                right = right[:-1]

            # bottom left corner
            if len(left) > 0 and equal(bottom[0].start, left[-1].end):
                bottom[0].start = left[-1].start
                bottom[0].mid = np.median(
                    np.array([bottom[0].start, bottom[0].end]), axis=0).astype(np.uint8)
                left = left[:-1]
        if len(top) > 0:
            # top right corner
            if len(right) > 0 and equal(top[-1].end, right[0].start):
                top[-1].end = right[0].end
                top[-1].mid = np.median(np.array([top[-1].start,
                                                  top[-1].end]), axis=0).astype(np.uint8)
                right = right[1:]

            # top left corner
            if len(left) > 0 and equal(top[0].start, left[0].start):
                top[0].start = left[0].end
                top[0].mid = np.median(
                    np.array([top[0].start, top[0].end]), axis=0).astype(np.uint8)
                left = left[1:]

        return top, bottom, right, left

    '''
    It will create a dictionary that will contain all of the boundaries for each side of the picture. It will load the top of the image, bottom of the image, the column of the left and the column of the right as a vector and count how paths are in those vectors using get_paths_boundaries
    '''

    def set_boundaries(self, sections_img):
        img = np.copy(sections_img)
        height, width = img.shape[0] - 1, img.shape[1] - 1
        top = [Boundary(coords, x=0)
               for coords in self.get_paths_boundaries(img[0, :])]
        bottom = [Boundary(coords, x=height)
                  for coords in self.get_paths_boundaries(img[-1, :])]
        right = [Boundary(coords, y=width)
                 for coords in self.get_paths_boundaries(img[:, -1])]
        left = [Boundary(coords, y=0)
                for coords in self.get_paths_boundaries(img[:, 0])]

        self.top, self.bottom, self.right, self.left = self.remove_one_in_the_corners(
            top, bottom, right, left)

        self.total = len(self.top) + len(self.bottom) + \
            len(self.right) + len(self.left)

    def get_active_lane(self):
        for b in self.bottom:
            if b.current_lane:
                return b
        return None

    def apply_offset(self, x, y):
        for bs in [self.top, self.bottom, self.right, self.left]:
            for b in bs:
                b.apply_offset(x, y)

    def get_boundaries_no_bottom(self):
        return [self.right] + [self.top] + [self.left]

    def empty(self):
        self.top, self.left, self.right, self.bottom = [], [], [], []
        self.total = 0

    '''
    It will return a list of string for every boundary in the image that contains 1 or more paths
    '''

    def __str__(self):
        boundaries = []
        ll = zip([self.top, self.bottom, self.right, self.left],
                 ["top", "bottom", "right", "left"])
        for v, s in ll:
            if len(v) > 0:
                boundaries.append("{}: {}".format(s, len(v)))
        return '  '.join(boundaries)

    def paint_boundaries_mid(self, img):
        for boundaries in [self.top, self.bottom, self.right, self.left]:
            for boundary in boundaries:
                if boundary.current_lane:
                    cv2.circle(img, tuple(boundary.mid), 2, [0, 145, 255], -1)
                else:
                    cv2.circle(img, tuple(boundary.mid), 2, [255, 0, 145], -1)
        return img


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

class Train:
    '''
    The purpose of this class is to train the model of the classifier. At the end of the constructor it will save the model in a file, so it won't be necessary to be creating the model every time.
    '''

    def __init__(self):
        self.originals_folder = "./images/train/originals"
        self.sections_folder = "./images/train/sections"

        data = self.get_training_data()
        self.train_model(data)

    def get_training_data(self):
        '''
        It will load all the normalized pixels and its labels that are available at ./images/train
        '''
        originals_files = glob.glob("{}/*".format(self.originals_folder))
        sections_files = glob.glob("{}/*".format(self.sections_folder))
        if len(originals_files) != len(sections_files):
            print("Check that the sections images and the originals images are the same")
        return open_images(list(zip(originals_files, sections_files)))

    def train_model(self, data):
        '''
        It will train the model and measure the time. The model will be saved.
        '''
        start = time.time()
        clf = EuclidianClassifier()
        clf.fit(data)
        dump(clf, './classifier-euclidian.joblib')
        end = time.time()
        print("Trained euclidian classifier in {} seconds".format(end - start))

class EuclidianClassifier():
    colors = {
            'b': [0, 255, 0],
            'l': [0, 0, 255],
            's': [255, 0, 0],
        }

    def __init__(self):
        self.centroids = np.array([])
        self.labels_centroids = np.array([])

    def normalize_img(self, img):
        return np.rollaxis((np.rollaxis(img, 2) + 0.0) / np.sum(img, 2), 0, 3)[:, :, :2]

    def get_pixels(self, data):
        '''
        Returns the normalized RGB values and its label as an array of tuples representing the images
        '''
        signs = []
        bck = []
        line = []

        for (original, labels) in data:
            normalized = self.normalize_img(original)
            signs = signs + [v for v in normalized[np.all(labels == [255, 0, 0], 2)]]
            bck = bck + [v for v in normalized[np.all(labels == [0, 255, 0], 2)]]
            line = line + [v for v in normalized[np.all(labels == [0, 0, 255], 2)]]
        normalized = signs + bck + line
        labels = ['s'] * len(signs) + ['b'] * len(bck) + ['l'] * len(line)

        return np.array(normalized), np.array(labels)
    
    def fit(self, data):
        '''
        It will train the model given a list of tuples. Each tuple in the list represent an image where the first value is the original frame and the second contains the labels
        '''
        X, labels = self.get_pixels(data)
        print("Training with...", 
            np.unique(labels, return_counts=True), "b: background, l: line, s:sign")

        self.labels_centroids = np.unique(labels)
        self.centroids = np.array([np.mean(X[labels == l], axis=0) for l in self.labels_centroids])
        # self.clf = SVC()
        # self.clf.fit(X, labels)

    def classes2sections(self, shape, classes):
        '''
        Converts the given matrix of labels to a image with the a color, for each pixel representing the class of the pixel (background=green, line=blue , or red=sign)
        '''

        labels = np.empty(classes.shape, dtype='<U1')
        labels[classes == 0] = self.labels_centroids[0]
        labels[classes == 1] = self.labels_centroids[1]
        labels[classes == 2] = self.labels_centroids[2]
        
        classes = classes.reshape(shape[:2])
        sections = np.empty(shape, dtype=np.uint8)
        sections[classes == 0] = self.colors[self.labels_centroids[0]]
        sections[classes == 1] = self.colors[self.labels_centroids[1]]
        sections[classes == 2] = self.colors[self.labels_centroids[2]]

        return sections, labels

    def labels2img(self, frame, labels):
        labels = labels.reshape(frame.shape[:2])
        img_out = np.empty(frame.shape, dtype=np.uint8)
        img_out[labels == 'b'] = [0, 255, 0]
        img_out[labels == 'l'] = [0, 0, 255]
        img_out[labels == 's'] = [255, 0, 0]
        return img_out

    def predict(self, X):
        '''
        Given an array of normalized pixels, it will return an array with the class of each pixel (a class can be 'b', 'l' or 's')
        '''
        predictions = np.linalg.norm(self.centroids[:, np.newaxis] - X, axis=2)
        classes = np.argmin(predictions, axis=0)
        return classes
        # return self.clf.predict(X)
    
    def predict_image(self, img):
        '''
        Given an image, it will return a matrix with the same shape but each pixel will have the correct color RGB
        '''
        X = self.normalize_img(img).reshape((-1, 2))
        classes = self.predict(X)
        sections, labels = self.classes2sections(img.shape, classes)
        # sections = self.labels2img(img, classes)
        return sections, classes
    
    def save_model():
        dump(self, './classifier.joblib')

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


class BrainFinalExam(Brain):
 
  NO_FORWARD = 0
  SLOW_FORWARD = 0.1
  MED_FORWARD = 0.5
  FULL_FORWARD = 1.0

  NO_TURN = 0
  MED_LEFT = 0.5
  HARD_LEFT = 1.0
  MED_RIGHT = -0.5
  HARD_RIGHT = -1.0

  NO_ERROR = 0
  N_FRAMES = 0

  MEMORY = []
  def setup(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/image", Image, self.callback)
    self.shape_model = TrainShape().train()

  def callback(self,data):
    self.rosImage = data

  def destroy(self):
    cv2.destroyAllWindows()

  def step(self):
	
    # access last image received from the camera and convert it into
    # opencv format
    try:
      self.cv_image = self.bridge.imgmsg_to_cv2(self.rosImage, "rgb8")
      # self.cv_image = self.bridge.imgmsg_to_cv2(self.rosImage, "rgb8")
    except CvBridgeError as e:
      print(e)

    # visualize the image
    cv2.imshow("Stage Camera Image", self.cv_image)
    cv2.waitKey(1)
    # write the image to a file, for debugging etc.
    # cv2.imwrite("test-file.jpg",self.cv_image)
    
    # TODO maybe not use clf as we already have colors classified???
    sections_img, labels = self.clf.predict_image(self.cv_image)
    sections_img = cv2.medianBlur(sections_img, 3)
    scene_description = SceneDescription(
        sections_img, MEMORY, w=60, h=60)
    scene_state = SceneState(
        scene_description, N_FRAMES, self.shape_model, self.debug_mode)
    MEMORY.append(scene_state)
    memory = memory[-120:]
    control = ControlCommand(memory)

    
    # Here you should process the image from the camera and calculate
    # your control variable(s), for now we will just give the controller
    # some 'fixed' values so that it will do something.
    lineDistance = .5 
    hasLine = 1

    # A trivial on-off controller
    if (hasLine):
      if (lineDistance > self.NO_ERROR):
        self.move(self.FULL_FORWARD,self.HARD_LEFT)
      elif (lineDistance < self.NO_ERROR):
        self.move(self.FULL_FORWARD,self.HARD_RIGHT)
      else:
        self.move(self.FULL_FORWARD,self.NO_TURN)
    else:
      # if we can't see the line we just stop, this isn't very smart
      self.move(self.NO_FORWARD, self.NO_TURN)
    
    N_FRAMES += 1
 
def INIT(engine):
  assert (engine.robot.requires("range-sensor") and
	  engine.robot.requires("continuous-movement"))

  # If we are allowed (for example you can't in a simulation), enable
  # the motors.
  try:
    engine.robot.position[0]._dev.enable(1)
  except AttributeError:
    pass

  return BrainFinalExam('BrainFinalExam', engine)
