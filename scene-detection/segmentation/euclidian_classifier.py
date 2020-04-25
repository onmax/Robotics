import numpy as np
from joblib import dump
import cv2
from sklearn.svm import SVC
from prepare_training.pixel_distribution import plot_distribution

class EuclidianClassifier():
    colors = {
            'b': [0, 255, 0],
            'l': [0, 0, 255],
            's': [255, 0, 0],
        }

    def __init__(self):
        self.centroids = np.array([])
        self.labels_centroids = np.array([])

    def normalize_img(self, img):
        return np.rollaxis((np.rollaxis(img, 2) + 0.0) / np.sum(img, 2), 0, 3)[:, :, :2]

    def get_pixels(self, data):
        '''
        Returns the normalized RGB values and its label as an array of tuples representing the images
        '''
        signs = []
        bck = []
        line = []

        for (original, labels) in data:
            normalized = self.normalize_img(original)
            signs = signs + [v for v in normalized[np.all(labels == [255, 0, 0], 2)]]
            bck = bck + [v for v in normalized[np.all(labels == [0, 255, 0], 2)]]
            line = line + [v for v in normalized[np.all(labels == [0, 0, 255], 2)]]
        normalized = signs + bck + line
        labels = ['s'] * len(signs) + ['b'] * len(bck) + ['l'] * len(line)

        return np.array(normalized), np.array(labels)
    
    def fit(self, data):
        '''
        It will train the model given a list of tuples. Each tuple in the list represent an image where the first value is the original frame and the second contains the labels
        '''
        X, labels = self.get_pixels(data)
        print("Training with...", 
            np.unique(labels, return_counts=True), "b: background, l: line, s:sign")

        self.labels_centroids = np.unique(labels)
        self.centroids = np.array([np.mean(X[labels == l], axis=0) for l in self.labels_centroids])
        # self.clf = SVC()
        # self.clf.fit(X, labels)

    def classes2sections(self, shape, classes):
        '''
        Converts the given matrix of labels to a image with the a color, for each pixel representing the class of the pixel (background=green, line=blue , or red=sign)
        '''

        labels = np.empty(classes.shape, dtype='<U1')
        labels[classes == 0] = self.labels_centroids[0]
        labels[classes == 1] = self.labels_centroids[1]
        labels[classes == 2] = self.labels_centroids[2]
        
        classes = classes.reshape(shape[:2])
        sections = np.empty(shape, dtype=np.uint8)
        sections[classes == 0] = self.colors[self.labels_centroids[0]]
        sections[classes == 1] = self.colors[self.labels_centroids[1]]
        sections[classes == 2] = self.colors[self.labels_centroids[2]]

        return sections, labels

    def labels2img(self, frame, labels):
        labels = labels.reshape(frame.shape[:2])
        img_out = np.empty(frame.shape, dtype=np.uint8)
        img_out[labels == 'b'] = [0, 255, 0]
        img_out[labels == 'l'] = [0, 0, 255]
        img_out[labels == 's'] = [255, 0, 0]
        return img_out

    def predict(self, X):
        '''
        Given an array of pixels normalized it will return an array with the class of each pixel (a class can be 'b', 'l' or 's')
        '''
        predictions = np.linalg.norm(self.centroids[:, np.newaxis] - X, axis=2)
        classes = np.argmin(predictions, axis=0)
        return classes
        # return self.clf.predict(X)
    
    def predict_image(self, img):
        '''
        Given an image, it will return a matrix with the same shape but each pixel will have the correct color RGB
        '''
        X = self.normalize_img(img).reshape((-1, 2))
        classes = self.predict(X)
        sections, labels = self.classes2sections(img.shape, classes)
        # sections = self.labels2img(img, classes)
        return sections, classes
    
    def save_model():
        dump(self, './classifier.joblib')
