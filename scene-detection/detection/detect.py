from joblib import load
import cv2
import time
from functions import *
from detection.boundaries import Boundaries
from detection.scene_moments import SceneMoments
from detection.scene_state import SceneState
from detection.control_command import ControlCommand

class Detect:
    def __init__(self, video_name, hide_preview, save_detection, debug_mode):
        self.hide_preview = hide_preview
        self.save_detection = save_detection
        self.debug_mode = debug_mode

        video_name = "video1.mp4" if video_name == None else video_name
        self.video_path = "./videos/input/{}".format(video_name)
        
        # Loads the classifier
        self.clf = load('./classifier-euclidian.joblib')
        print("Detecting using the video {}".format(self.video_path))
        
        self.detect_video()
    
    def detect_video(self):
        start = time.time()
        n_frames = self.detect_video_loop()
        end = time.time()
        predicted_in(start, end, n_frames)

    def detect_video_loop(self):
        memory = []

        n_frames = 0
        cap = cv2.VideoCapture(self.video_path)
        cap.set(1, 620) # 2550

        if self.save_detection:
            video_name = self.video_path.split('/')[-1]
            filename = "./videos/output/" + video_name.split('.')[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filename, fourcc, 30, (320 * 4, 240 * 4))

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                sections_img, labels = self.clf.predict_image(frame)
                # median filter
                sections_img = cv2.medianBlur(sections_img,5)
                boundaries = Boundaries(sections_img)
                scene_moments_line = SceneMoments(sections_img, [255, 0, 0], type_object="line")
                scene_moments_signs = SceneMoments(sections_img, [0, 0, 255], min_contour_size=1000, type_object="sign")


                scene_state = SceneState(sections_img, boundaries, scene_moments_line, scene_moments_signs, n_frames, self.debug_mode)
                memory.append(scene_state)

                decision = ControlCommand(memory).decision
                if self.debug_mode:
                    sections_img = scene_moments_line.paint_contours(sections_img, [255, 150, 36])
                    sections_img = scene_moments_line.paint_lines(sections_img, [255, 255, 0])
                    sections_img = scene_moments_line.paint_defects(sections_img, [255, 0, 255])
                    sections_img = scene_moments_signs.paint_lines(sections_img, [0, 255, 255])
                    sections_img = scene_moments_signs.paint_defects(sections_img, [0, 120, 255])
                    sections_img = boundaries.paint_boundaries_mid(sections_img)
                    text = [str(boundaries), ""] + scene_moments_line.sstr() + scene_moments_signs.sstr() + scene_state.sstr() + [decision]
                else:
                    text = [decision]



                sections_img = cv2.resize(sections_img, (sections_img.shape[1] * 4, sections_img.shape[0] * 4))
                sections_img = write_text(sections_img, text)
                
                if not self.hide_preview:
                    cv2.imshow("Images", sections_img)
                    cv2.waitKey(0)
                
                if self.save_detection:
                    out.write(sections_img)

                n_frames += 1

            else:
                cap.release()
                break
        if self.save_detection:
            out.release()

        cv2.destroyAllWindows()

        return n_frames