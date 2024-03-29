import numpy as np
import imageio
import cv2
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import time

def normalized_img(a):
    return np.rollaxis((np.rollaxis(a, 2) + 0.0) / np.sum(a, 2), 0, 3)[:, :, :2]


def img2labels(shape, section):
    labels = np.empty(shape, dtype=np.str)
    labels[np.all(section == [0, 255, 0], 2)] = 'background'
    labels[np.all(section == [255, 0, 0], 2)] = 'line'
    labels[np.all(section == [0, 0, 255], 2)] = 'symbol'
    return labels


def labels2img(frame, labels):
    labels = labels.reshape(frame.shape[:2])
    img_out = np.empty(frame.shape, dtype=np.uint8)
    img_out[labels == 'b'] = [0, 255, 0]
    img_out[labels == 'l'] = [0, 0, 255]
    img_out[labels == 's'] = [255, 0, 0]
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
    return img_out

def labelsN2img(frame, labels):
    labels = labels.reshape(frame.shape[:2])
    img_out = np.empty(frame.shape, dtype=np.uint8)
    labels = labels.astype('<U1')
    img_out[np.where(labels == '0')] = [0, 255, 0]
    img_out[np.where(labels == '1')] = [0, 0, 255]
    img_out[np.where(labels == '2')] = [255, 0, 0]
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
    return img_out

def open_images(indexes):
    '''
    Return a dictionary containing:
        - image: An array of all the pixels.
            The shape of the array is: len(frame_numbers) x width x height x 3
        - section: An array of all the different sections. Only contains one of the following colors:
            - Red: symbol
            - Green: background
            - Blue: line.
            The shape of the array is: len(frame_numbers) x width x height x 3
        normalized: An array of all the pixels
            The shape of the array is: len(frame_numbers) x width x height x 2
        labels: An array of all the pixels labeled
            It will contain: 'b', 'l' or 's'
            The shape of the array is: len(frame_numbers) x width x height x 1
    '''
    print("Reading images...")

    d = {
        'image': np.array([], dtype=np.uint8),
        'section': np.array([], dtype=np.uint8),
        'normalized': np.array([], dtype=np.float),
        'labels': np.array([], dtype=np.str)
    }

    for i in indexes:
        image = np.array(imageio.imread(
            "./images/originals/frame-{}.png".format(i)))
        section = np.array(imageio.imread(
            "./images/sections/frame-{}.png".format(i)))

        d["image"] = np.append(d["image"], image)
        d["section"] = np.append(d["section"], section)
        d["normalized"] = np.append(
            d["normalized"], normalized_img(image))
        d["labels"] = np.append(
            d["labels"], img2labels(image.shape[:2], section))

    d["image"] = d["image"].reshape((-1, 3))
    d["section"] = d["section"].reshape((-1, 3))
    d["normalized"] = d["normalized"].reshape((-1, 2))
    return d

def predicted_in(start, n_frames):
    end = time.time()
    n_seconds = end - start
    print("Predicted in {} seconds {} frames. That is {} seconds/frame".format(
        n_seconds, n_frames, n_seconds / n_frames))

def plot(pixels, sections):
    data_symbol = pixels[np.where(
        np.all(np.equal(sections, (255, 0, 0)), 1))]
    data_background = pixels[np.where(
        np.all(np.equal(sections, (0, 255, 0)), 1))]
    data_lines = pixels[np.where(
        np.all(np.equal(sections, (0, 0, 255)), 1))]

    plt.figure()
    plt.plot(data_symbol[:, 0], data_symbol[:, 1], 'r.', label='Symbol')
    plt.plot(data_background[:, 0],
             data_background[:, 1], 'g.', label='Background')
    plt.plot(data_lines[:, 0], data_lines[:, 1], 'b.', label='Lines')
    plt.title('Normalized RGB')

    plt.show()
