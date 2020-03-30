import time
import cv2
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import numpy as np
start = time.time()

capture = cv2.VideoCapture("../input/video1.mp4")

lower_blue = np.array([90, 60, 60])
upper_blue = np.array([255, 255, 255])
lower_red1 = np.array([0, 65, 75])
upper_red1 = np.array([12, 255, 255])
lower_red2 = np.array([240, 65, 75])
upper_red2 = np.array([255, 255, 255])

WIDTH = 320
HEIGHT = 240

name = 'segmentation.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(name, fourcc, 30, (WIDTH * 2, HEIGHT * 2))


while capture.isOpened():
    black_picture = np.zeros([HEIGHT, WIDTH,3],dtype=np.uint8)
    ret, frame = capture.read()
    if ret != True:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask_line = cv2.inRange(hsv, lower_blue, upper_blue)
    res_line = cv2.bitwise_and(frame, frame, mask=mask_line)
    black_picture[mask_line == 255] = [255, 0, 0]

    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_arrow = cv2.add(mask_r1, mask_r2)
    res_arrow = cv2.bitwise_and(frame, frame, mask=mask_arrow)
    black_picture[mask_arrow == 255] = [0, 0, 255]


    top = np.concatenate((frame, black_picture), axis=1)
    bottom = np.concatenate((res_line, res_arrow), axis=1)
    total = np.concatenate((top, bottom), axis=0)
    out.write(total)
    cv2.imshow('total', total)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()

end = time.time()
