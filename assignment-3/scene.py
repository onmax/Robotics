from joblib import load
import cv2
from functions import *
import time
import os
import cv2

class Scene:
    def __init__(self):
        super().__init__()

        self.video_name = "video1.mp4"
        clf_name = "euclidian"
        self.clf = load('../models/classifier-{}.joblib'.format(clf_name))
        self.save_video()
        print("Finished")


    def save_video(self):

        path = "../output"
        os.makedirs(path)

        start = time.time()
        n_frames = 0

        filename = "../output/" + self.video_name.split('.')[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 30, (320 * 4, 240 * 4))

        capture = cv2.VideoCapture("../input/" + self.video_name)
        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                img_out = get_segmented_img(self.clf, frame)
                # img_out = apply_gaussian_filter(img_out)
                # remove blur
                boundaries = detect_boundaries(img_out)
                texts, frame = get_scene_context(img_out, frame)
                frame = cv2.resize(frame, (frame.shape[1] * 4, frame.shape[0] * 4))
                frame = write_text(frame, texts + boundaries)
                out.write(frame)
                n_frames += 1

            else:
                capture.release()
                break
        out.release()
        cv2.destroyAllWindows()

        if n_frames == 0:
            print("No frames processed...")
        else:
            predicted_in(start, n_frames)

