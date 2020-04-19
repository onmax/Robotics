import sys
from prepare_training.prepare_training import PrepareTraining
from train.train import Train
from detection.detect import Detect

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-pt", "--prepare_training",
                    help="The program will prepare the images for training",action="store_true", default=False)
parser.add_argument("--frames_from_video", type=str,
                    help="Set the filename that the program will show in order to choose the frames the classifier will use as training")
parser.add_argument("--remove_training_frames",
                    help="It will remove all the frames stored in ./images/train/originals.", action="store_true", default=False)
parser.add_argument("--paint_frames",
                    help="It will allow the user to paint all of the frames stored in the ./images/train/originals folder.", action="store_true", default=False)
parser.add_argument("-t", "--train",
                    help="It will train the model using the images stored in the ./images/train/{originals, sections} folder.", action="store_true", default=False)
parser.add_argument("-d", "--detect",
                    help="It will detect the different components in the image from the video. If --input_video is not defined it will use video1.mp4", action="store_true", default=False)
parser.add_argument("-i", "--input_video", type=str,
                    help="It will allow to the user to specify the video which it will be classify and detected. The video must be in the folder ./videos/input/")
parser.add_argument("-s", "--save_detection", type=str, # TODO
                    help="It will save the video with the output of the detection algorithm in ./videos/output/")
parser.add_argument("-hp", "--hide_preview", # TODO
                    help="It will hide the preview, so showing the preview in the window won't affect to the measurement of the time.")
args = parser.parse_args()

# If no arguments are given, show help message
if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

# Prepare training
if args.prepare_training:
    PrepareTraining(paint_frames=args.paint_frames, video=args.frames_from_video, remove_frames=args.remove_training_frames)


# Train model
if args.train:
    Train()

# Get the confusion matrix of the model
if "--stats-model" in sys.argv:
    pass

# Given a video it detects the diferent elements
if args.detect:
    Detect(video_name=args.input_video)