
import numpy as np
import preprocessing
import postprocessing
import matplotlib.pyplot as plt
import sklearn, sklearn.tree, sklearn.model_selection, sklearn.ensemble

ftcount = 531

datafile = 'Dataset/dataset.train'

train = preprocessing.get_data(datafile, ftcount)
train = preprocessing.remove_data(train)
train = preprocessing.remove_features(train)

trainm = preprocessing.mask_unused_features(train)
        
x = []
meany = []
sdy = []
t = 7
N=100
results = np.zeros((N,1))
for i in range(N):
    print("{0:.2%}".format(float(i)/N))
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=50)

    cv_rf = sklearn.model_selection.cross_val_score(rf, trainm[:,:-1], trainm[:,-1], cv=t)
    results[i] = cv_rf.mean()
plt.show()

