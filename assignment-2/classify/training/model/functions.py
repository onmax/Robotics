import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
import time


def normalized_img(a):
    return np.rollaxis((np.rollaxis(a, 2)+0.0)/np.sum(a, 2), 0, 3)[:, :, :2]


def get_distances(section):
    new_image = np.ones(section.shape[:2], np.uint8)
    new_image[np.where(np.all(np.equal(section, (0, 255, 0)), 2))] = 0
    dist = cv2.distanceTransform(new_image, cv2.DIST_L2, 3)
    return dist


def plot_rgb(bck, line, signs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(signs[:, 0], signs[:, 1], signs[:, 2], 'r.', label='marca')
    ax.plot(bck[:, 0], bck[:, 1], bck[:, 2], 'g.', label='fondo')
    ax.plot(line[:, 0], line[:, 1], line[:, 2], 'b.', label='linea')

    ax.set_xlabel('RGB normalized 1')
    ax.set_ylabel('RGB normalized 2')
    ax.set_zlabel('Distance')

    ax.legend()
    plt.title('Distribution of RGB and distances')
    plt.show()


def get_probabilities(shape, input_prob):
    ff = np.full(shape, False)
    prob = np.random.rand(shape)
    ff[np.where(prob < input_prob)] = False
    ff[np.where(prob >= input_prob)] = True
    return ff, prob


def get_matrices(normalized, section, distances, size):
    sign = normalized[np.where(
        np.all(np.equal(section, (255, 0, 0)), 2))]
    sign_distance = distances[np.where(
        np.all(np.equal(section, (255, 0, 0)), 2))]
    sign = np.insert(sign, 2, sign_distance, axis=1)

    bck = normalized[np.where(
        np.all(np.equal(section, (0, 255, 0)), 2))]
    bck_distance = distances[np.where(
        np.all(np.equal(section, (0, 255, 0)), 2))]
    bck = np.insert(bck, 2, bck_distance, axis=1)

    line = normalized[np.where(
        np.all(np.equal(section, (0, 0, 255)), 2))]
    line_distance = distances[np.where(
        np.all(np.equal(section, (0, 0, 255)), 2))]
    line = np.insert(line, 2, line_distance, axis=1)

    sign, bck, line = remove_random_pixels(
        sign, bck, line, size)
    return sign, bck, line


def remove_random_pixels(sign, bck, line, size):
    ff, prob = get_probabilities(
        bck.shape[0], 1 - (size / bck.shape[0]) * 1.4)  # 0.9997
    bck = bck[np.where(ff)]

    # remove useless some blue
    if line.shape[0] != 0:
        ff, prob = get_probabilities(
            line.shape[0], 1 - (size / line.shape[0]))  # 0.997
        line = line[np.where(ff)]

    if sign.shape[0] != 0:
        ff, prob = get_probabilities(
            sign.shape[0], 1 - (size / sign.shape[0]) * 0.05)  # 0.5
        sign = sign[np.where(ff)]

    return sign, bck, line


def get_values_cm(confusion_matrix, real, prediction, section, index):
    if real[1] == 255:
        real = 'bck'
    elif real[2] == 255:
        real = 'line'
    elif real[0] == 255:
        real = 'sign'

    if real == prediction and real == section:
        confusion_matrix["tp"] += 1
    elif real == prediction and real != section:
        confusion_matrix["tn"] += 1
    elif real != prediction and prediction == section:
        confusion_matrix["fp"] += 1
    elif real != prediction and prediction != section:
        confusion_matrix["fn"] += 1
    return confusion_matrix


def get_confusion_matrix(real_data, predictions):
    [cm_bck, cm_lines, cm_sign] = [{"tp": 0, "tn": 0, "fp": 0, "fn": 0}] * 3
    for (real, prediction) in zip(real_data, predictions):
        cm_bck = get_values_cm(cm_bck, real, prediction, 'bck', 1)
        cm_lines = get_values_cm(cm_bck, real, prediction, 'line', 2)
        cm_sign = get_values_cm(cm_bck, real, prediction, 'sign', 0)
    return {"bck": cm_bck, "lines": cm_lines, "sign": cm_sign}


def predict(clf, data):
    return clf.predict(data)


def get_all_confusion_matrixes(test_indexes, clf):
    confusion_matrices = []
    times = []
    for (index, i) in enumerate(test_indexes):
        print("Testing: {}%".format(index * 100 / len(test_indexes)))
        image = np.array(imageio.imread(
            "D:/proyectos/robotics/assignment-2/classify/training/images/originals/frame-{}.png".format(i)))
        section = np.array(imageio.imread(
            "D:/proyectos/robotics/assignment-2/classify/training/images/sections/frame-{}.png".format(i)))

        normalized = normalized_img(image)

        data = normalized.reshape(-1, 2)

        start = time.time()
        predictions = predict(clf, data)
        end = time.time()
        times.append(end - start)

        confusion_matrix = get_confusion_matrix(
            section.reshape(-1, 3), predictions)
        confusion_matrices.append(confusion_matrix)
    return confusion_matrices, np.array(times)


def get_stats(cms, times):
    gen_confusion_matrix = {
        "bck": {
            "tp": sum(x["bck"]["tp"] for x in cms),
            "tn": sum(x["bck"]["tn"] for x in cms),
            "fp": sum(x["bck"]["fp"] for x in cms),
            "fn": sum(x["bck"]["fn"] for x in cms)
        },
        "line": {
            "tp": sum(x["lines"]["tp"] for x in cms),
            "tn": sum(x["lines"]["tn"] for x in cms),
            "fp": sum(x["lines"]["fp"] for x in cms),
            "fn": sum(x["lines"]["fn"] for x in cms)
        },
        "sign": {
            "tp": sum(x["sign"]["tp"] for x in cms),
            "tn": sum(x["sign"]["tn"] for x in cms),
            "fp": sum(x["sign"]["fp"] for x in cms),
            "fn": sum(x["sign"]["fn"] for x in cms)
        }
    }
    gen_confusion_matrix["all"] = {
        "tp": sum(gen_confusion_matrix[x]['tp'] for x in gen_confusion_matrix),
        "tn": sum(gen_confusion_matrix[x]['tn'] for x in gen_confusion_matrix),
        "fp": sum(gen_confusion_matrix[x]['fp'] for x in gen_confusion_matrix),
        "fn": sum(gen_confusion_matrix[x]['fn'] for x in gen_confusion_matrix)
    }

    cm = gen_confusion_matrix["all"]
    lines = gen_confusion_matrix["line"]
    print("Mean time: {}".format(times.mean()))
    print("Overall precision", (cm['tp'] + cm['tn']) /
          (cm['tp'] + cm['tn'] + cm['fp'] + cm['fn']))
    print("Lines precision", (lines['tp'] + lines['tn']) /
          (lines['tp'] + lines['tn'] + lines['fp'] + lines['fn']))
