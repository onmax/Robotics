import numpy as np
from joblib import dump
from sklearn import svm
import time
from functions import open_images
from sklearn import discriminant_analysis as da


class Model:
    def __init__(self):
        super().__init__()
        self.set_dfs()
        self.train_model()
        dump(self.clf, './classifier.joblib')

    def set_dfs(self):
        frame_numbers = np.array(
            [12, 25, 35, 36, 39, 46, 53, 71, 82, 108, 112, 121, 125, 137, 158, 170, 181])
        self.train_df = open_images(frame_numbers)
    
    def feature_selection(data):
        pass

    def train_model(self):
        start = time.time()

        pixels_normalized = self.train_df["normalized"]
        labels = self.train_df["labels"]

        # self.clf = svm.SVC()
        # self.clf.fit(pixels_normalized, labels)

        X_train = self.scaler.fit_transform(pixels_normalized, labels)
        X_test = self.scaler.transform(X_test)

        self.clf = da.QuadraticDiscriminantAnalysis().fit(pixels_normalized, labels)

        end = time.time()
        print("Trained in {} seconds".format(end - start))
