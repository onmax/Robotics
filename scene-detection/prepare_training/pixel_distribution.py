import numpy as np
from matplotlib import pyplot as plt

def plot_distribution(pixels, sections):
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