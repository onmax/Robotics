from joblib import load
import cv2
import time
from functions import *
from detection.boundaries import Boundaries
from detection.scene_moments import SceneMoments

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
                boundaries = Boundaries(sections_img)
                scene_context_line = SceneMoments(sections_img, [255, 0, 0], type_object="line")
                scene_context_signs = SceneMoments(sections_img, [0, 0, 255], min_contour_size=400, type_object="sign")
                text = [str(boundaries), ""] + scene_context_line.sstr() + scene_context_signs.sstr()

                sections_img = scene_context_line.paint_lines(sections_img, [255, 255, 0])
                sections_img = scene_context_line.paint_defects(sections_img, [255, 0, 255])
                sections_img = scene_context_signs.paint_lines(sections_img, [0, 255, 255])
                sections_img = scene_context_signs.paint_defects(sections_img, [0, 120, 255])

                sections_img = cv2.resize(sections_img, (sections_img.shape[1] * 4, sections_img.shape[0] * 4))
                sections_img = write_text(sections_img, text)
                cv2.imshow("Images", sections_img)
                cv2.waitKey(0)
                # n_frames += 1

            else:
                cap.release()
                break
        cv2.destroyAllWindows()

        if n_frames == 0:
            print("No frames processed...")
        else:
            predicted_in(start, n_frames)
