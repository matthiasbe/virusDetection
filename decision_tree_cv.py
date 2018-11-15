
import numpy as np
import preprocessing
import postprocessing
import matplotlib.pyplot as plt
import sklearn, sklearn.tree, sklearn.model_selection, sklearn.ensemble, sklearn.utils

ftcount = 531

datafile = 'Dataset/dataset.train'

train = preprocessing.get_data(datafile, ftcount)

trainm = preprocessing.mask_unused_features(train)

x = []
meany = []
sdy = []

for t in range(2,10):
    results = []
    for i in range(1,1000):
        clf = sklearn.tree.DecisionTreeClassifier()
        shuffled = sklearn.utils.shuffle(trainm[:,:-1])
        cv_dt = sklearn.model_selection.cross_val_score(clf, shuffled, trainm[:,-1], cv=t)
        results.append(np.average(cv_dt))

    x.append(t)
    meany.append(np.mean(results))
    sdy.append(np.std(results))
    print(t)


np.savetxt("results/decision_tree.txt",[x,meany,sdy].T);

plt.subplot(121)
plt.plot(x, meany)
plt.title("Moyenne")

plt.subplot(122)
plt.plot(x, sdy)
plt.title("Standard deviation")

plt.show()
