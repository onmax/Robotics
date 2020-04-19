import numpy as np
from joblib import dump
import glob
import time
from train.euclidian_classifier import EuclidianClassifier
from functions import *


class Train:
    '''
    The purpose of this class is to train the model of the classifier. At the end of the constructor it will save the model in a file, so it won't be necessary to be creating the model every time.
    '''

    def __init__(self):
        self.originals_folder = "./images/train/originals"
        self.sections_folder = "./images/train/sections"

        data = self.get_training_data()
        self.train_model(data)

    def get_training_data(self):
        '''
        It will load all the normalized pixels and its labels that are available at ./images/train
        '''
        originals_files = glob.glob("{}/*".format(self.originals_folder))
        sections_files = glob.glob("{}/*".format(self.sections_folder))
        if len(originals_files) != len(sections_files):
            print("Check that the sections images and the originals images are the same")
        return open_images(list(zip(originals_files, sections_files)))

    def train_model(self, data):
        '''
        It will train the model and measure the time. The model will be saved.
        '''
        start = time.time()
        clf = EuclidianClassifier()
        clf.fit(data)
        dump(clf, './classifier-euclidian.joblib')
        end = time.time()
        print("Trained euclidian classifier in {} seconds".format(end - start))