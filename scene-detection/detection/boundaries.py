import numpy as np
import cv2
from functions import r, g, b


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
