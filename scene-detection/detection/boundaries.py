import numpy as np

class Boundaries:
    def __init__(self, sections_img):
        self.set_boundaries(sections_img)
    
    '''
    This function wil read how many changes are in a array
    i.e. => [0, 255, 255, 255, 0, 255, 255, 0]
              ^              ^  ^         ^
    There are 4 changes => 2 paths. If the last and the first are differents is also a change.
    '''
    def get_paths_boundaries(self, x):
        return int((np.diff(x[np.any(x != [0, 0, 255], -1)], axis=0) == 255).sum() / 2)
    
    '''
    It will create a dictionary that will contain all of the boundaries for each side of the picture. It will load the top of the image, bottom of the image, the column of the left and the column of the right as a vector and count how paths are in those vectors using get_paths_boundaries
    '''
    def set_boundaries(self, sections_img):
        self.top = self.get_paths_boundaries(sections_img[0,:])
        self.bottom = self.get_paths_boundaries(sections_img[-1,:])
        self.right = self.get_paths_boundaries(sections_img[:, -1])
        self.left = self.get_paths_boundaries(sections_img[:,0])

    '''
    It will return a list of string for every boundary in the image that contains 1 or more paths
    '''
    def __str__(self):
        boundaries = []
        ll = zip([self.top, self.bottom, self.right, self.left], ["top", "bottom", "right", "left"])
        for v, s in ll:
            if v > 0:
                boundaries.append("{}: {}".format(s, v))
        return '  '.join(boundaries)