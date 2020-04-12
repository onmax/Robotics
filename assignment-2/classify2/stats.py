import numpy as np
from sklearn.metrics import confusion_matrix
from joblib import load
from functions import *
import time


class Stats:
    def __init__(self):
        super().__init__()
        self.set_dfs()
        plot(self.test_df["normalized"], self.test_df["section"])

        clfs = [
            # load('./models/classifier-kneighbors.joblib'),
            # load('./models/classifier-mlpclass.joblib'),
            # load('./models/classifier-qda.joblib'),
            # load('./models/classifier-decisiontree.joblib'),
            # load('./models/classifier-randomforest.joblib'),
            # load('./models/classifier-adaboost.joblib')
            # load('./models/classifier-gaussian.joblib'),
            load('./models/classifier-euclidian.joblib')
            ]
        for i, clf in enumerate(clfs):
            print("CLF", i)
            self.test_model(clf)
            print("\n\n")

    def set_dfs(self):
        self.frame_numbers = np.arange(1, 186)
        self.test_df = open_images(self.frame_numbers)

    def test_model(self, clf):
        print("Predicting...")
        start = time.time()

        pixels_normalized = self.test_df["normalized"]

        predicted = clf.predict(pixels_normalized)

        n_frames = len(self.frame_numbers)
        predicted_in(start, n_frames)

        expected = self.test_df["labels"]
        results = confusion_matrix(expected, predicted, np.unique(expected))

        index = [np.where(np.unique(expected) == 'b'),
                 np.where(np.unique(expected) == 'l'),
                 np.where(np.unique(expected) == 's')]

        def precission(i):
            values = results[index[i]].flatten()
            prob = (values[index[i]] / sum(values))[0]
            return round(prob * 100, 5)

        print("Full confusion matrix\n", results)
        print("Precission:")
        print("Background: {}%".format(precission(0)))
        print("Symbol: {}%".format(precission(2)))
        print("Line: {}%".format(precission(1)))
