from joblib import load
import cv2
from functions import *
import time
import os
import shutil
import cv2

class Predict:
    def __init__(self):
        super().__init__()

        self.video_name = "video1.mp4"
        clf_name = "euclidian"
        self.clf = load('./models/classifier-{}.joblib'.format(clf_name))
        # self.save_video()
        self.save_frames(clf_name)
        print("Finished")

    def save_frames(self, clf_name):
        start = time.time()
        n_frames = 0

        path = "../output/{}".format(clf_name)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        picture_index = 1
        
        capture = cv2.VideoCapture("../input/" + self.video_name)
        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                normalized = normalized_img(frame)
                labels = self.clf.predict(normalized.reshape((-1, 2)))
                img_out = labels2img(frame, labels)
                cv2.imwrite("{}/frame-{:d}.jpg".format(path, picture_index), img_out)
                picture_index += 1
                n_frames += 1
            else:
                capture.release()
                break
        cv2.destroyAllWindows()

        predicted_in(start, n_frames)


    def save_video(self):
        start = time.time()
        n_frames = 0

        filename = "../output/" + self.video_name.split('.')[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 30, (320, 240))

        capture = cv2.VideoCapture("../input/" + self.video_name)
        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                normalized = normalized_img(frame)
                labels = self.clf.predict(normalized.reshape((-1, 2)))
                img_out = labels2img(frame, labels)
                out.write(img_out)
                n_frames += 1
            else:
                capture.release()
                break
        out.release()
        cv2.destroyAllWindows()

        predicted_in(end, n_frames)

