import numpy as np
import imageio
import cv2
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import time
import copy

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

def get_segmented_img(clf, frame):
    normalized = normalized_img(frame)
    labels = clf.predict(normalized.reshape((-1, 2)))
    img_out = labels2img(frame, labels)
    return img_out

def apply_gaussian_filter(img):
    return cv2.GaussianBlur(img, (13, 13),0)


def detect_curve(contour, defects, i=0):
    # This funciton could be improved
    s = np.array([contour[defects[i][0]]])[0][0][0]
    e = np.array([contour[defects[i][1]]])[0][0][0]
    m = np.array([contour[defects[i][2]]])[0][0][0]
    direction = "derecha" if e - s < m else "izquierda"
    return direction

def distance(p1, p2):
    return (sum([(a - b) ** 2 for a, b in zip(p1, p2)]))**0.5

def get_paths_boundaries(x):
    # This function wil read how many changes are in a array
    # i.e. => [0, 255, 255, 255, 0, 255, 255, 0]
    #           ^              ^  ^         ^
    # There are 4 changes => 2 paths. If the last and the first are differents is also a change.
    # print( (np.diff(x, axis=0) == 255))
    return int((np.diff(x[np.any(x != [0, 0, 255], -1)], axis=0) == 255).sum() / 2)

def detect_closeness(contour, defects):
    # Gets the distance between the three points
    mids = [np.array([contour[d[2]]])[0][0] for d in defects]
    distances = np.array([distance(p1, p2) for i, p1 in enumerate(mids) for p2 in mids[i+1:]])

    # Number of curves
    curves = [i for i, dist in enumerate(distances) if dist > 50]
    n_curves = len(curves)

    n_in_a_cross = len(distances) - n_curves
    
    # A point can only be in a cross or curve. So, if the points does not belong to a
    # cross, then it is a curve 
    return n_curves, n_in_a_cross, curves

def get_scene_context(img, frame):

    paths = {
        "Arriba": get_paths_boundaries(img[0,:]),
        "Abajo": get_paths_boundaries(img[-1,:]),
        "Derecha": get_paths_boundaries(img[:,-1]),
        "Izquierda": get_paths_boundaries(img[:,0])
    }
    
    boundaries = []
    for p in paths:
        if paths[p] > 0:
            boundaries.append("{}: {}".format(p, paths[p]))

    img_bw = np.all(img == [255, 0, 0], axis=-1).astype(np.uint8)[90:,:] * 255
    contours, _  = cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return ["Nada"], frame

    chull_list = [cv2.convexHull(contour,returnPoints=False) for contour in contours]
    all_defects = [cv2.convexityDefects(contour, chull) for (contour,chull) in zip(contours, chull_list)]

    if len(all_defects) == 0 or np.all(np.equal(all_defects[0], None)):
        return "Recto", frame

    contour = contours[0]
    defects = all_defects[0]
    
    defects = defects[:,0,:].tolist()
    defects =[[start, end, mid, length] for start, end, mid, length in defects if length > 1000]

    # Print contours and holes in picture
    for s, e, m, l in defects:
        cv2.line(img, tuple(contour[s][0]), tuple(contour[e][0]), (120,0,120), 2)
        cv2.circle(frame, (contour[m][0][0], contour[m][0][1] + 90), 5,[120,120,255],-1)

    if len(defects) == 0:
        text = ["Recto"]
    elif len(defects) == 1:
        direction = detect_curve(contour, defects)
        text = ["Curva " + direction]
    elif len(defects) == 2:
        n_curves = detect_closeness(contour, defects)
        if n_curves == 2:
            direction1 = detect_curve(contour, defects, 0)
            direction2 = detect_curve(contour, defects, 1)
            text = ["Curva " + direction1, "Curva " + direction2]
        else:
            text = ["Cruce 2 salidas"]
    elif len(defects) == 3:
        n_curves, n_in_a_cross, curves_indexes = detect_closeness(contour, defects)
        if n_curves == 3:
            direction1 = detect_curve(contour, defects, 0)
            direction2 = detect_curve(contour, defects, 1)
            direction3 = detect_curve(contour, defects, 2)
            text = ["Curva " + direction1, "Curva " + direction2, "Curva " + direction3]
        elif n_curves == 1:
            direction = detect_curve(contour, defects, curves_indexes[0])
            text = ["Cruce dos salidas", "Curva " + direction]
        else:
            text = ["Cruce dos salidas"]
    else:
        text = ["Cruce tres salidas"]
    return text + boundaries, frame


def write_text(img, texts):
    # https://gist.github.com/aplz/fd34707deffb208f367808aade7e5d5c
    bck = (0, 0, 0)
    color = (255, 255, 255)
    
    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_scale = 0.7
    texts_sizes = [cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0] for text in texts]

    text_width, text_height = max([s[0] for s in texts_sizes]), sum([s[1] for s in texts_sizes]) + 10 * (len(texts) - 1)

    padding = 6
    box_coords = ((0,0), (text_width + padding, text_height + padding + 15))

    img = cv2.rectangle(img, box_coords[0], box_coords[1], bck, cv2.FILLED)
    
    for i, text in enumerate(texts):
        padding_top = 0 if i == 0 else sum(s[1] for s in texts_sizes[:i]) + 10 * i
        img = cv2.putText(img, text, (padding, padding + 15 + padding_top), font, fontScale=font_scale, color=color, thickness=1)

    return img
   


