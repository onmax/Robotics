import sys
from model import Model
from stats import Stats
from predict import Predict

if "--model" in sys.argv:
    Model()

if "--stats" in sys.argv:
    Stats()

if "--predict" in sys.argv:
    Predict()
