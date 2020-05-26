from joblib import load
import cv2
import time
from functions import *
from detection.scene_description import SceneDescription
from detection.scene_state import SceneState
from detection.control_command import ControlCommand
from shape_detection.train_shape import TrainShape


class Detect:
    def __init__(self, video_name, hide_preview, save_detection, debug_mode, segmented_background):
        self.hide_preview = hide_preview
        self.save_detection = save_detection
        self.debug_mode = debug_mode
        self.segmented_background = segmented_background

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
        model = TrainShape().train()

        memory = []

        n_frames = 0
        cap = cv2.VideoCapture(self.video_path)
        cap.set(1, 450)  # 1250 2550, 1500

        if self.save_detection:
            video_name = self.video_path.split('/')[-1]
            filename = "./videos/output/" + video_name.split('.')[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filename, fourcc, 15, (320 * 4, 240 * 4))

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                sections_img, labels = self.clf.predict_image(frame)
                sections_img = cv2.medianBlur(sections_img, 3)
                scene_description = SceneDescription(
                    sections_img, memory, w=60, h=60)
                scene_state = SceneState(
                    scene_description, n_frames, model, self.debug_mode)
                memory.append(scene_state)
                memory = memory[-120:]
                control = ControlCommand(memory)

                output_image = sections_img if self.segmented_background else frame

                if self.debug_mode:
                    output_image = scene_description.paint_verbose(
                        output_image)
                    output_image = control.paint_vector(output_image)
                    text = scene_description.sstr() + scene_state.sstr() + control.sstr()
                else:
                    text = scene_state.sstr() + control.sstr()
                    output_image = control.paint_vector(output_image)

                output_image = cv2.resize(
                    output_image, (output_image.shape[1] * 4, output_image.shape[0] * 4))
                output_image = write_text(output_image, text)
                if not self.hide_preview:
                    cv2.imshow("Images", output_image)
                    cv2.waitKey(1)

                if self.save_detection:
                    out.write(output_image)

                n_frames += 1

            else:
                cap.release()
                break
        if self.save_detection:
            print("Video saved at", filename)
            out.release()

        cv2.destroyAllWindows()

        return n_frames
