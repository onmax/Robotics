from joblib import load
import cv2
from functions import *


class Predict:
    def __init__(self):
        super().__init__()

        self.video_name = "video1.mp4"
        self.clf = load('./classifier.joblib')
        self.save_video()
        print("Finished")

    def save_video(self):
        filename = "../output/" + self.video_name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 30, (320, 240))
        capture = cv2.VideoCapture("../input/" + self.video_name)
        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                normalized = frame
                labels = self.clf.predict(normalized.reshape((-1, 2)))
                img_out = labels2img(frame, labels)
                out.write(img_out)
            else:
                capture.release()
                break
        out.release()
        cv2.destroyAllWindows()
