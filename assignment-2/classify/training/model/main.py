import numpy as np
from sklearn import svm
from functions import *
from dataframe import *
import time
from joblib import dump

WIDTH = 320
HEIGHT = 240

TEST_2D = True
TEST_3D = False


frame_numbers = np.array(
    [12, 25, 35, 36, 39, 46, 53, 71, 82, 108, 112, 121, 125, 137, 158, 170, 181])
np.random.shuffle(frame_numbers)
training_indexes = frame_numbers[:int(len(frame_numbers) * 0.7)]
test_indexes = frame_numbers[int(len(frame_numbers) * 0.7):]
test_indexes = np.arange(160, 186)

TRAINING_SIZE = int(200 / (len(frame_numbers) * 3))

pixels = np.array([])
labels = np.array([])

df = set_dataframe(training_indexes)

backgrounds = []
lines = []
signs = []
# remove large amount of data and just store useful data

# for each image we need:
for index, row in df.iterrows():
    sign, bck, line = get_matrices(
        row.normalized, row.section, row.distances, TEST_3D, TRAINING_SIZE)
    backgrounds.append(bck)
    lines.append(line)
    signs.append(sign)

# flatten one level
backgrounds = np.array([x for y in backgrounds for x in y])
lines = np.array([x for y in lines for x in y])
signs = np.array([x for y in signs for x in y])

print("Training model with the following data: \n{} pixels for background\n{} pixels for lines\n{} pixels for signs".format(
    len(backgrounds), len(lines), len(signs)))

if TEST_3D:
    time3d_start = time.time()
    clf3d = svm.SVC()
    clf3d.fit(np.concatenate([backgrounds, lines, signs]), [
        "bck"] * backgrounds.shape[0] + ["line"] * lines.shape[0] + ["sign"] * signs.shape[0])
    time3d_end = time.time()
    print("Trained 3D model in {}".format(time3d_end - time3d_start))

if TEST_2D:
    time2d_start = time.time()
    clf2d = svm.SVC()
    clf2d.fit(np.concatenate([backgrounds[:, :2], lines[:, :2], signs[:, :2]]), [
        "bck"] * backgrounds.shape[0] + ["line"] * lines.shape[0] + ["sign"] * signs.shape[0])
    time2d_end = time.time()
    print("Trained 2D model in {}".format(time2d_end - time2d_start))

print("Getting accuracy with {} images".format(len(test_indexes)))
if TEST_2D:
    cms, times2d = get_all_confusion_matrixes(
        test_indexes, clf2d, False)
    get_stats(cms, times2d)
    # save the classifier
    dump(clf2d, './assignment-2/classify/training/model/classifier2d.joblib')
if TEST_3D:
    cms, times3d = get_all_confusion_matrixes(
        test_indexes, clf3d, True)
    get_stats(cms, times3d)
    dump(clf2d, './assignment-2/classify/training/model/classifier3d.joblib')
