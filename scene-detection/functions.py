import numpy as np
import imageio
import cv2


b, g, r = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
top_offset = 30

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


def predicted_in(start, end, n_frames):
    n_seconds = end - start
    print("Predicted in {} seconds {} frames. That is {} seconds/frame".format(
        n_seconds, n_frames, n_seconds / n_frames))
        
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
   