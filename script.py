import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import sklearn
import preprocessing
import postprocessing
import vis

ftcount = 531

datafile = 'Dataset/dataset.train'

train = preprocessing.get_data(datafile, ftcount)

trainm = preprocessing.mask_unused_features(train)
#train_masked = train[mask]
#print(train_masked.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
trainm_train, trainm_test = train_test_split(trainm, train_size=0.9, test_size=0.1)
clf = LogisticRegression(solver='lbfgs', max_iter=1000).fit(trainm_train[:,:-1], trainm_train[:,-1])
clf2 = svm.SVC(gamma=0.001, C=100).fit(trainm_train[:,:-1], trainm_train[:,-1])
print(clf.score(trainm_test[:,:-1], trainm_test[:,-1]))
print(clf2.score(trainm_test[:,:-1], trainm_test[:,-1]))
#virus_proportion = (np.sum(trainm[:,-1])/trainm.shape[0])
#print(virus_proportion)
#clf.predict(trainm_test[:,:-1])