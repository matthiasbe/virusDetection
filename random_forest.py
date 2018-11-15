
import numpy as np
import preprocessing
import postprocessing
import matplotlib.pyplot as plt
import sklearn, sklearn.tree, sklearn.model_selection, sklearn.ensemble

ftcount = 531

datafile = 'Dataset/dataset.train'

train = preprocessing.get_data(datafile, ftcount)

trainm = preprocessing.mask_unused_features(train)


x = []
meany = []

for t in range(5,8):
    results = []
        
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=50, max_depth=5)

    shuffled = sklearn.utils.shuffle(trainm[:,:-1])
    cv_rf = sklearn.model_selection.cross_val_score(rf, shuffled, trainm[:,-1], cv=t)
    result = np.average(cv_rf)
    x.append(t)
    meany.append(result)
    print(str(t) + " : " + str(result))

np.savetxt("results/random_forest.txt",np.array([x,meany]).T);

plt.plot(x, meany)
plt.title("Moyenne")


plt.show()

