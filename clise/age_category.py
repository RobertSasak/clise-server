# import modules
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn import decomposition
from sklearn.utils.extmath import fast_dot

# load classifier
clf_filename = 'clise/classifier.joblib.pkl'
clf = joblib.load(clf_filename)

# predict function
def predict(data):
    data = data.values.reshape(1, -1)
    return clf.predict(data).tolist()
