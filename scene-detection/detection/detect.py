from joblib import load
import cv2
import time
from functions import *

class Detect:
    def __init__(self, video_name):
        video_name = "video1.mp4" if video_name == None else video_name
        self.video_path = "./videos/input/{}".format(video_name)
        
        # Loads the classifier
        self.clf = load('./classifier-euclidian.joblib')
        print("Detecting using the video {}".format(self.video_path))
        
        self.detect_video()
    
    def detect_video(self):
        start = time.time()
        self.detect_video_loop()
        end = time.time()
        n_seconds = end - start
        print("Predicted in {} seconds TODO frames. That is TODO seconds/frame".format(n_seconds))

    def detect_video_loop(self):
        n_frames = 0
        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                sections_img, labels = self.clf.predict_image(frame)
                print(labels)
                cv2.imshow("Images", sections_img)
                cv2.waitKey(0)
                # boundaries = detect_boundaries(img_sections)
                # texts, frame = get_scene_context(img_sections, frame)
                # frame = cv2.resize(frame, (frame.shape[1] * 4, frame.shape[0] * 4))
                # frame = write_text(frame, texts + boundaries)
                # n_frames += 1

            else:
                capture.release()
                break
        cv2.destroyAllWindows()

        if n_frames == 0:
            print("No frames processed...")
        else:
            predicted_in(start, n_frames)
