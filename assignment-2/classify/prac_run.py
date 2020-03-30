import cv2
from matplotlib import pyplot as plt
import numpy as np
from joblib import load

capture = cv2.VideoCapture("./assignment-2/input/video1.mp4")

WIDTH = 320
HEIGHT = 240

cap = cv2.VideoCapture(
    './assignment-2/input/{}'.format("video1.mp4"))
count = 1

lower_blue = np.array([90, 60, 60])
upper_blue = np.array([255, 255, 255])
lower_red1 = np.array([0, 65, 75])
upper_red1 = np.array([12, 255, 255])
lower_red2 = np.array([240, 65, 75])
upper_red2 = np.array([255, 255, 255])


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


def normalized_img(a):
    return np.rollaxis((np.rollaxis(a, 2)+0.0)/np.sum(a, 2), 0, 3)[:, :, :2]


c = 0
name = 'segmentation.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(name, fourcc, 30, (WIDTH, HEIGHT))
clf = load('./assignment-2/classify/training/model/classifier2d.joblib')
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        normalized = normalized_img(frame)
        labels = clf.predict(normalized.reshape((-1, 2)))

        labels_image = []
        for label in labels:
            if label == 'bck':
                labels_image.append([0, 255, 0])
            elif label == 'line':
                labels_image.append([0, 0, 255])
            elif label == 'sign':
                labels_image.append([255, 0, 0])
        img_sections = np.array(
            labels_image, dtype=np.uint8).reshape((WIDTH, HEIGHT, 3))
        out.write(img_sections)
        c += 1
        cap.set(1, c * 25)
    else:
        cap.release()
        break
out.release()
cv2.destroyAllWindows()
