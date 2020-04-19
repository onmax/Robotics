import numpy as np
import imageio
import cv2


def open_images(images_paths):
    '''
    Returns a list of tuples containing (original_image, image_labelled)
    '''
    data = []

    for (path_original, path_train) in images_paths:
        print("Reading images:", path_train, path_original)

        original = np.array(imageio.imread(path_original))
        label = np.array(imageio.imread(path_train))
        data.append((original, label))

    return data



def get_sections_img(clf, frame):
    '''
    Returns and image of same size of frame where every pixel is classify
    in one of the possible classes (background=green, line=blue or 
    red=sign)
    '''
    normalized = normalized_img(frame)
    labels = clf.predict(normalized)
    color_labels = labels2sections(frame, labels)
    return color_labels