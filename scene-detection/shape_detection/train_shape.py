import cv2
import imageio
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut


class TrainShape():
    def __init__(self):
        img_raw_list, labels = self.get_img_n_labels()
        self.k_fold(img_raw_list, labels)
    
    def get_img_n_labels(self):
        img_names = sorted(glob.glob('../images/shape/*.png'))
        img_raw_list = np.array([imageio.imread(image) for image in img_names])
        labels = np.array([x.split("/")[-1].split("-")[0] for x in img_names])
        return img_raw_list, labels
        
    def k_fold(self, X, y):
        loo = LeaveOneOut()
        predictions = []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index][0]
            y_train, y_test = y[train_index], y[test_index][0]
            clf = KneighboursClassifier().fit(X_train, y_train)
            predictions.append(clf.predict(X_test) == y_test)
        print(np.unique(predictions, return_counts=True))
        

class KneighboursClassifier():
    def fit(self, img_raw_list, labels):
        X = []
        for img_raw in img_raw_list:
            X.append(self.contour2des(img_raw, self.img2contour(img_raw)))
        X = np.array(X).reshape((-1, 32)).astype(np.uint8)
        self.neigh = KNeighborsClassifier(n_neighbors=5, metric=self.hamming_dist)
        self.neigh.fit(X, labels)  #X son los datos de entrenamiento y son los datos objetivo
        return self

    def predict(self, img):
        X = self.contour2des(img, self.img2contour(img))
        return self.neigh.predict(X)
    
    def hamming_dist(self, d1, d2):
        d1, d2 = d1.astype(np.uint8), d2.astype(np.uint8)
        assert d1.dtype == np.uint8 and d2.dtype == np.uint8
        d1_bits = np.unpackbits(d1)
        d2_bits = np.unpackbits(d2)
        return np.bitwise_xor(d1_bits, d2_bits).sum()

    def img2contour(self, img):
        img = img[:,:,:3]
        bw = np.all(img == [255, 0, 0], axis=-1).astype(np.uint8) * 255
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        return contours[0]

    def contour2des(self, img, contour):
        (x, y), axis, angle = cv2.fitEllipse(contour)
        axis = np.array(axis)
        if angle > 90:
            angle -= 180
        orb = cv2.ORB_create()
        kp = cv2.KeyPoint(x, y, np.mean(axis) * 1.3, angle - 90)
        lkp, des = orb.compute(img, [kp])
        return des

TrainShape()
