import numpy as np
from sklearn.metrics import confusion_matrix
from joblib import load
from functions import open_images, plot
import time


class Stats:
    def __init__(self):
        super().__init__()
        self.set_dfs()
        plot(self.test_df["normalized"], self.test_df["section"])

        self.clf = load('./classifier.joblib')
        self.test_model()

    def set_dfs(self):
        print("Reading images...")
        self.frame_numbers = np.arange(70, 186)
        self.test_df = open_images(self.frame_numbers)

    def test_model(self):
        print("Predicting...")
        start = time.time()

        pixels_normalized = self.test_df["normalized"]

        predicted = self.clf.predict(pixels_normalized)

        end = time.time()

        n_frames = len(self.frame_numbers)
        n_seconds = end - start
        print("Predicted in {} seconds {} frames. That is {} seconds/frame".format(
            n_seconds, n_frames, n_seconds / n_frames))

        expected = self.test_df["labels"]
        results = confusion_matrix(expected, predicted, np.unique(expected))

        index = [np.where(np.unique(expected) == 'b'),
                 np.where(np.unique(expected) == 's'),
                 np.where(np.unique(expected) == 'l')]

        def precission(i):
            values = results[index[i]].flatten()
            prob = (values[index[i]] / sum(values))[0]
            return round(prob * 100, 5)

        print("Full confusion matrix\n", results)
        print("Precission:")
        print("Background: {}%".format(precission(0)))
        print("Symbol: {}%".format(precission(1)))
        print("Line: {}%".format(precission(2)))
