
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

for t in range(2,11):
    results = []
        
x = []
meany = []
sdy = []

for t in range(2,11):
    results = []
    for i in range(1,100):
        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=50, max_depth=5)

        cv_rf = sklearn.model_selection.cross_val_score(rf, trainm[:,:-1], trainm[:,-1], cv=t)
        results.append(np.average(cv_rf))
    
    x.append(t)
    meany.append(np.mean(results))
    sdy.append(np.std(results))
    print(t)


np.savetxt("results/random_forest.txt",np.array([x,meany,sdy]).T);

plt.subplot(121)
plt.plot(x, meany)
plt.title("Moyenne")

plt.subplot(122)
plt.plot(x, sdy)
plt.title("Standard deviation")




plt.show()

