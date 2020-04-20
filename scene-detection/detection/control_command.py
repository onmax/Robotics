'''
- Command control for the line.
    1. if number of contours in complementary is 0 or 1 => Nothing
    2. if number of contours in complementary is 2 => Straight line or curve
        2.1 if number of defects is 0 => Straight line
        2.2 else => curve
            2.2.2 If the sum of boundaries is 2, and at least one of them is from the botton and the other one is going to on of the sides. If the side is left => curve left, or if it is right => curve right
            2.2.1 Fit a ellipsis in the contour and get the value of inclination. If angle is negative => curve left. If angle is positive => curve right
    3. If number of contours in complementary is 3 => three-way street
    4. If number of contours in complementary is 4 or more => four-way street. There is no five-way street or more.
'''

class ItemsOnTheWay():
    # TODO try to remove this constructor
    def __init__(self):
        pass

    def set_params(self, is_straight_line=False, curve_direction=None, n_way_stree=None):
        self.is_straight_line = is_straight_line
        self.curve_direction = curve_direction
        self.n_way_stree = n_way_stree

    def nothing(self):
        self.set_params()
    
    def set_straight_line(self):
        self.set_params(is_straight_line=True)


class ControlCommand()
    def __init__(self, boundaries, sc_line, sc_sign):
        # Items on the way
        self.it_way = self.detect_items_on_the_way(boundaries, sc_line)

    def detect_items_on_the_way(self, boundaries, sc_line):
        # step 1
        if len(sc_line.contours_compl) <= 1:
            return ItemsOnTheWay.nothing()
        
        # step 2
        if len(sc_line.contours_compl) <= 2:
            if len(sc_line.defects) == 0:
                return ItemsOnTheWay.straight_line()
            else:
                pass





