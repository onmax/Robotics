from joblib import load
import cv2
from functions import *
import time
import os
import shutil
import cv2

class Segmentation:
    def __init__(self):
        super().__init__()

        self.video_name = "video1.mp4"
        clf_name = "euclidian"
        self.clf = load('../models/classifier-{}.joblib'.format(clf_name))
        self.save_video()
        print("Finished")


    def save_video(self):
        N_FRAMES = 400000 # Just to speed up the process. Should be removed

        #TEMP
        save_frame = False
        if save_frame:
            path = "../output/frames/"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

        start = time.time()
        n_frames = 0

        filename = "../output/" + self.video_name.split('.')[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 30, (320 * 4, 240 * 4))

        capture = cv2.VideoCapture("../input/" + self.video_name)
        while capture.isOpened():
            # 1240, 240, 0, 1720, 800, 600, 1300, 1500
            # capture.set(1,1895); 
            one_frame = False
            ret, frame = capture.read()
            if ret:
                img_out = get_segmented_img(self.clf, frame)
                # img_out = apply_gaussian_filter(img_out)
                # remove blur
                texts, frame = get_scene_context(img_out, frame)
                frame = cv2.resize(frame, (frame.shape[1] * 4, frame.shape[0] * 4))
                frame = write_text(frame, texts)
                out.write(frame)
                if save_frame:
                    cv2.imwrite("{}/frame-{:d}.jpg".format(path, n_frames), frame)
                n_frames += 1

                
                # Should be removed
                if one_frame or n_frames == N_FRAMES:
                    capture.release()
                    break
            else:
                capture.release()
                break
        out.release()
        cv2.destroyAllWindows()

        if n_frames == 0:
            print("No frames processed...")
        else:
            predicted_in(start, n_frames)

