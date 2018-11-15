
import numpy as np
import preprocessing
import postprocessing
import matplotlib.pyplot as plt
import sklearn, sklearn.tree, sklearn.model_selection, sklearn.ensemble

ftcount = 531

datafile = 'Dataset/dataset.train'

train = preprocessing.get_data(datafile, ftcount)

trainm = preprocessing.mask_unused_features(train)



rf = sklearn.ensemble.RandomForestClassifier(n_estimators=50)

shuffled = sklearn.utils.shuffle(trainm[:,:-1])
cv_rf = sklearn.model_selection.cross_val_score(rf, shuffled, trainm[:,-1], cv=8)
print(np.average(cv_rf))


