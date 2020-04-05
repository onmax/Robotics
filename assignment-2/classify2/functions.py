import numpy as np
import imageio
import cv2
from sklearn.preprocessing import StandardScaler


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
    d = {
        'image': np.array([], dtype=np.uint8),
        'section': np.array([], dtype=np.uint8),
        'normalized': np.array([], dtype=np.float),
        'labels': np.array([], dtype=np.str)
    }

    for i in indexes:
        image = np.array(imageio.imread(
            "../images/originals/frame-{}.png".format(i)))
        section = np.array(imageio.imread(
            "../images/sections/frame-{}.png".format(i)))

        d["image"] = np.append(d["image"], image)
        d["section"] = np.append(d["section"], section)
        d["normalized"] = np.append(
            d["normalized"], normalized_img(image))
        d["labels"] = np.append(
            d["labels"], img2labels(image.shape[:2], section))

    d["image"] = d["image"].reshape((-1, 3))
    d["section"] = d["section"].reshape((-1, 3))
    d["normalized"] = d["normalized"].reshape((-1, 2))

    scaler = StandardScaler()
    d["normalized"] = d["normalized"].reshape((-1, 2))
    d["normalized"] = scaler.fit_transform(d["normalized"])

    return d
