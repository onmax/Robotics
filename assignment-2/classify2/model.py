import numpy as np
from joblib import dump
from sklearn import svm
import time
from sklearn.datasets import make_classification
import cv2

from functions import open_images
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


class EuclidianClassifier():
    centroids = np.array([])
    labels_centroids = np.array([])

    def __init__(self):
        pass

    def fit(self, X, y):
        self.labels_centroids = np.unique(y)
        self.centroids = np.array([np.mean(X[y == label], axis=0) for label in self.labels_centroids])

    def predict(self, X):
        predictions = np.linalg.norm(self.centroids[:, np.newaxis] - X, axis=2)
        classes = np.argmin(predictions, axis=0)
        
        # Numbers to labels and to shape image
        labels = np.empty(classes.shape, dtype='<U1')
        labels[classes == 0] = self.labels_centroids[0]
        labels[classes == 1] = self.labels_centroids[1]
        labels[classes == 2] = self.labels_centroids[2]
        return labels

class Model:
    '''
    The purpose of this class is to train the model of the classifier. At the end of the constructor it will save the model in a file, so it won't be necessary to be creating the model every time.
    '''

    def __init__(self):
        super().__init__()
        self.set_dfs()
        self.train_model()

    def set_dfs(self):
        '''
        Set the dataframes. It loads the frames within the folder image with the following indexes and for each of them will load.
        See open_images doc to see the return type
        '''
        # frame_numbers = np.array(
        # [12, 25, 35, 36, 39, 46, 53, 71, 82, 108, 112, 121, 125, 137, 158, 170, 181])
        frame_numbers = np.arange(1, 186)
        self.train_df = open_images(frame_numbers)

    def feature_selection(data):
        pass

    def train_model(self):
        print("Training...")

        pixels_normalized = self.train_df["normalized"]
        labels = self.train_df["labels"]

        # self.clf = svm.SVC()
        # self.clf.fit(pixels_normalized, labels)

        # self.clf = KMeans(n_clusters=3, random_state=0).fit(pixels_normalized, labels)
        # self.clf = da.(n_clusters=3, random_state=0).fit(pixels_normalized, labels)

        # start = time.time()
        # clf = KNeighborsClassifier(n_neighbors=3)
        # clf.fit(pixels_normalized, labels)
        # dump(clf, './models/classifier-kneighbors.joblib')
        # end = time.time()
        # print("Trained KNeighborsClassifier in {} seconds".format(end - start))

        # start = time.time()
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, 4), random_state=1)
        # clf.fit(pixels_normalized, labels)
        # dump(clf, './models/classifier-mlpclass.joblib')
        # end = time.time()
        # print("Trained mlpclass in {} seconds".format(end - start))

        # start = time.time()
        # clf = QuadraticDiscriminantAnalysis()
        # clf.fit(pixels_normalized, labels)
        # dump(clf, './models/classifier-qda.joblib')
        # end = time.time()
        # print("Trained qda in {} seconds".format(end - start))

        # start = time.time()
        # clf = DecisionTreeClassifier(max_depth=5).fit(pixels_normalized, labels)
        # dump(clf, './models/classifier-decisiontree.joblib')
        # end = time.time()
        # print("Trained decision tree in {} seconds".format(end - start))

        # Trained Random forest in 527.401358127594 seconds
        # start = time.time()
        # clf = RandomForestClassifier(max_depth=5).fit(pixels_normalized, labels)
        # dump(clf, './models/classifier-randomforest.joblib')
        # end = time.time()
        # print("Trained Random forest in {} seconds".format(end - start))

        # start = time.time()
        # clf = AdaBoostClassifier().fit(pixels_normalized, labels)
        # dump(clf, './models/classifier-adaboost.joblib')
        # end = time.time()
        # print("Trained Ada boost in {} seconds".format(end - start))
        
        # start = time.time()
        # clf = GaussianNB().fit(pixels_normalized, labels)
        # dump(clf, './models/classifier-gaussian.joblib')
        # end = time.time()
        # print("Trained Gaussian in {} seconds".format(end - start))

        start = time.time()
        clf = EuclidianClassifier()
        clf.fit(pixels_normalized, labels)
        dump(clf, './models/classifier-euclidian.joblib')
        end = time.time()
        print("Trained euclidian in {} seconds".format(end - start))