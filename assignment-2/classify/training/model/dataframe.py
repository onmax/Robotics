import pandas as pd
import numpy as np
import imageio
from functions import *


def set_dataframe(training_indexes):
    d = {'image': [], 'section': [], 'normalized': [], 'distances': []}

    for i in training_indexes:
        image = np.array(imageio.imread(
            "D:/proyectos/robotics/assignment-2/classify/training/images/originals/frame-{}.png".format(i)))
        section = np.array(imageio.imread(
            "D:/proyectos/robotics/assignment-2/classify/training/images/sections/frame-{}.png".format(i)))

        d["image"].append(image)
        d["section"].append(section)
        d["normalized"].append(normalized_img(image))
        d["distances"].append(get_distances(section))

    df = pd.DataFrame(data=d)
    df.head()

    return df
