
import numpy as np
import preprocessing
import postprocessing
import matplotlib.pyplot as plt
import sklearn, sklearn.tree, sklearn.model_selection, sklearn.ensemble

ftcount = 531

datafile = 'Dataset/dataset.train'

train = preprocessing.get_data(datafile, ftcount)

trainm = preprocessing.mask_unused_features(train)

test_size=10
train_data, test_data = sklearn.model_selection.train_test_split(trainm, test_size=test_size)

clf = sklearn.tree.DecisionTreeClassifier()
clf.fit(train_data[:,:-1],train_data[:,-1])
res = clf.predict(test_data[:,:-1])
precision = 1-sum([abs(i)/2 for i in (res - test_data[:,-1])])/test_size

print(precision)

# cv_dt = sklearn.model_selection.cross_val_score(clf, trainm[:,:-1], trainm[:,-1], cv=10)
# np.average(cv_dt)