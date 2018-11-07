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
data = list(csv.reader(open(datafile), delimiter=' '))
train = np.zeros((len(data),ftcount))

# converts "15:1" to row[15] = 1
idx = 0
for row in data:
    current = train[idx]
    current[ftcount-1] = int(row[0])
    for elt in row[1:]:
        split = elt.split(':')
        if (len(split) == 2):
            ftnum = int(split[0])
            if (ftnum <= len(current)-1):
                current[ftnum] = int(split[1])
    idx = idx + 1

print(train.shape)
mask = np.zeros_like(train)
mask[:] = 1
empty_columns = train.any(axis=0)
mask[:, empty_columns] = 0
trainm = np.ma.masked_array(train, mask)
trainm
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