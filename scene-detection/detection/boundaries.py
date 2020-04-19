class Boundaries:
    def __init__(self):
        self.set_boundaries()
    
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
        self.boundaries = {
            "top": self.get_paths_boundaries(sections_img[0,:]),
            "bottom": self.get_paths_boundaries(sections_img[-1,:]),
            "right": self.get_paths_boundaries(sections_img[:,-1]),
            "left": self.get_paths_boundaries(sections_img[:,0])
        }

    '''
    It will return a list of string for every boundary in the image that contains 1 or more paths
    '''
    def boundaries_str(self):
        boundaries = []
        for p in self.boundaries:
            if paths[p] > 0:
                boundaries.append("{}: {}".format(p, paths[p]))
        return boundaries