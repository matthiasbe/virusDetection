
import numpy as np
import preprocessing
import postprocessing
import matplotlib.pyplot as plt
import sklearn, sklearn.tree, sklearn.model_selection, sklearn.ensemble

ftcount = 531

datafile = 'Dataset/dataset.train'

train = preprocessing.get_data(datafile, ftcount)

trainm = preprocessing.mask_unused_features(train)

test_size=100
results = []
for i in range(1,100):
    train_data, test_data = sklearn.model_selection.train_test_split(trainm, test_size=test_size)

    clf = sklearn.tree.DecisionTreeClassifier()
    clf.fit(train_data[:,:-1],train_data[:,-1])
    res = clf.predict(test_data[:,:-1])

    precision = 1-sum([abs(i)/2 for i in (res - test_data[:,-1])])/test_size
    results.append(precision)


plt.subplot(131)
plt.hist(results, bins=10, range=[0.5,1])
plt.title("1000 iterations : 100-large test sample")
plt.xlabel("Precision")

test_size=50
results = []
for i in range(1,100):
    train_data, test_data = sklearn.model_selection.train_test_split(trainm, test_size=test_size)

    clf = sklearn.tree.DecisionTreeClassifier()
    clf.fit(train_data[:,:-1],train_data[:,-1])
    res = clf.predict(test_data[:,:-1])

    precision = 1-sum([abs(i)/2 for i in (res - test_data[:,-1])])/test_size
    results.append(precision)


plt.subplot(132)
plt.hist(results, bins=10, range=[0.5,1])
plt.title("1000 iterations : 50-large test sample")
plt.xlabel("Precision")

test_size=30
results = []
for i in range(1,100):
    train_data, test_data = sklearn.model_selection.train_test_split(trainm, test_size=test_size)

    clf = sklearn.tree.DecisionTreeClassifier()
    clf.fit(train_data[:,:-1],train_data[:,-1])
    res = clf.predict(test_data[:,:-1])

    precision = 1-sum([abs(i)/2 for i in (res - test_data[:,-1])])/test_size
    results.append(precision)


plt.subplot(133)
plt.hist(results, bins=10, range=[0.5,1])
plt.title("1000 iterations : 30-large test sample")
plt.xlabel("Precision")
plt.show()

# cv_dt = sklearn.model_selection.cross_val_score(clf, trainm[:,:-1], trainm[:,-1], cv=10)
# np.average(cv_dt)
