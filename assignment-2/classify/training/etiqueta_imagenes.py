import cv2
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import select_pixels as sel
import numpy as np
import copy

lower_blue = np.array([90, 60, 60])
upper_blue = np.array([255, 255, 255])
lower_red1 = np.array([0, 65, 75])
upper_red1 = np.array([12, 255, 255])
lower_red2 = np.array([240, 65, 75])
upper_red2 = np.array([255, 255, 255])

WIDTH = 320
HEIGHT = 240


def get_sections(frame):
    green_bck = np.zeros([HEIGHT, WIDTH, 3], dtype=np.uint8)
    green_bck[:] = [0, 255, 0]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask_line = cv2.inRange(hsv, lower_blue, upper_blue)
    res_line = cv2.bitwise_and(frame, frame, mask=mask_line)
    green_bck[mask_line == 255] = [0, 0, 255]

    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_arrow = cv2.add(mask_r1, mask_r2)
    res_arrow = cv2.bitwise_and(frame, frame, mask=mask_arrow)
    green_bck[mask_arrow == 255] = [255, 0, 0]

    return green_bck


'''
Get all frames
'''
videos = [
    {
        "name": "video1",
        "extension": "mp4"
    },
    {
        "name": "video2",
        "extension": "mp4"
    },
    {
        "name": "video3",
        "extension": "avi"
    },
    {
        "name": "video4",
        "extension": "avi"
    }
]

picture_index = 1

for video in videos:
    cap = cv2.VideoCapture(
        '../../input/{}.{}'.format(video["name"], video["extension"]))
    count = 1

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            sections = get_sections(copy.deepcopy(frame))
            image_supervised, finish = sel.select_fg_bg(sections)

            if finish:
                break

            imsave('./images/originals/frame-{:d}.png'.format(
                picture_index), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imsave(
                './images/sections/frame-{:d}.png'.format(picture_index), image_supervised)
            count += 1
            picture_index += 1
            cap.set(1, count * 50)
        else:
            cap.release()
            break
