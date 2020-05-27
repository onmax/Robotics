import cv2
import imageio
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
import time

class TrainShape():
    def get_img_n_labels(self):
        img_names = sorted(
            glob.glob('./images/shape/*.png'))
        img_raw_list = np.array([imageio.imread(image) for image in img_names])
        labels = np.array([x.split("/")[-1].split("-")[0] for x in img_names])
        return img_raw_list, labels

    def k_fold(self):
        X, y = self.get_img_n_labels()
        loo = LeaveOneOut()
        predictions = []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index][0]
            y_train, y_test = y[train_index], y[test_index][0]
            start = time.time()
            clf = KneighboursClassifier().fit(X_train, y_train)
            predictions.append(clf.predict(X_test) == y_test)
            print(clf.predict(X_test), y_test)
            end = time.time()
            print("average time", (end - start) / len(y)**2)
        print(np.unique(predictions, return_counts=True))
    def train(self):
        img_raw_list, labels = self.get_img_n_labels()
        return KneighboursClassifier().fit(img_raw_list, labels)

class KneighboursClassifier():
    def fit(self, img_raw_list, labels):
        X = []
        for img_raw in img_raw_list:
            X.append(self.contour2des(img_raw, self.img2contour(img_raw)))
        X = np.array(X).reshape((-1, 32)).astype(np.uint8)
        self.neigh = KNeighborsClassifier(
            n_neighbors=5, metric=self.hamming_dist)
        # X son los datos de entrenamiento y son los datos objetivo
        self.neigh.fit(X, labels)
        return self

    def predict(self, img):
        X = self.contour2des(img, self.img2contour(img))
        if np.all(X == None):
            return []
        else:
            return self.neigh.predict(X)

    # Función usada para calcular la distancia entre dos vectores
    def hamming_dist(self, d1, d2):
        d1, d2 = d1.astype(np.uint8), d2.astype(np.uint8)
        assert d1.dtype == np.uint8 and d2.dtype == np.uint8
        d1_bits = np.unpackbits(d1)
        d2_bits = np.unpackbits(d2)
        return np.bitwise_xor(d1_bits, d2_bits).sum()

    # Convertimos la imagen a contorno
    def img2contour(self, img):
        img = img[:, :, :3]
        bw = np.logical_and(img[:, :, 0] >= 230, img[:, :, 1]
                            < 20, img[:, :, 2] < 20).astype(np.uint8) * 255
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        return contours[0] if len(contours) > 0 else None

    # Convertimos el contorno a un descriptor ORB
    def contour2des(self, img, contour):
        (x, y), axis, angle = cv2.fitEllipse(contour)
        axis = np.array(axis)
        if angle > 90:
            angle -= 180
        orb = cv2.ORB_create()
        kp = cv2.KeyPoint(x, y, np.mean(axis) * 1.3, angle - 90)
        lkp, des = orb.compute(img, [kp])
        return des
