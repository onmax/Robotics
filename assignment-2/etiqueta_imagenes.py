import time
import cv2
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import select_pixels as sel
import numpy as np


# Abres el video / camara con

capture = cv2.VideoCapture("./video1.mp4")

# Lees las imagenes y las muestras para elegir la(s) de entrenamiento
# posibles funciones a usar


# cv2.imshow("Frame", frame)
# capture.release()
# cv2.destroyWindow("Frame")

# Si deseas mostrar la imagen con funciones de matplotlib posiblemente haya que cambiar
# el formato, con
# cv2.cvtColor(frame, code=CV_RGB2GRAY)

# Esta funcion del paquete "select_pixels" pinta los pixeles en la imagen
# Puede ser util para el entrenamiento

# markImg = sel.select_fg_bg(frame)

# Tambien puedes mostrar imagenes con las funciones de matplotlib
# plt.imshow(markImg)
# plt.show()

# Si deseas guardar alguna imagen ....

# imsave('lineaMarcada.png', markImg)

# The order of the colors is blue, green, red
while capture.isOpened():
    ret, frame = capture.read()

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([255, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('original', frame)
    cv2.imshow('Gray', gray)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
