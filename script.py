import pandas as pd
import csv
import matplotlib.pyplot as plt
import sklearn
import preprocessing
import postprocessing
import vis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm

ftcount = 531

datafile = 'Dataset/dataset.train'

train = preprocessing.get_data(datafile, ftcount)

trainm = preprocessing.mask_unused_features(train)

train, test = train_test_split(trainm, train_size=0.9, test_size=0.1)
clf = LogisticRegression(solver='lbfgs', max_iter=1000).fit(train[:,:-1], train[:,-1])
prediction = clf.predict(test[:,:-1])
postprocessing.evaluate(prediction, test[:,-1])
#clf2 = svm.SVC(gamma=0.001, C=100).fit(trainm_train[:,:-1], trainm_train[:,-1])
#print(clf.score(trainm_test[:,:-1], trainm_test[:,-1]))
#print(clf2.score(trainm_test[:,:-1], trainm_test[:,-1]))
#virus_proportion = (np.sum(trainm[:,-1])/trainm.shape[0])
#print(virus_proportion)
#clf.predict(trainm_test[:,:-1])