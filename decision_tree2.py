
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
sdy = []

for t in range(20,340,20):
    test_size = t
    results = []
    for i in range(1,1000):
        train_data, test_data = sklearn.model_selection.train_test_split(trainm, test_size=test_size)

        clf = sklearn.tree.DecisionTreeClassifier()
        clf.fit(train_data[:,:-1],train_data[:,-1])
        res = clf.predict(test_data[:,:-1])

        precision = 1-sum([abs(i)/2 for i in (res - test_data[:,-1])])/test_size
        results.append(precision)

    x.append(t)
    meany.append(np.mean(results))
    sdy.append(np.std(results))
    print(t)

np.savetxt("results/decision_tree.txt",np.array([x,meany,sdy]).T);

plt.subplot(121)
plt.plot(x, meany)
plt.title("Moyenne")

plt.subplot(122)
plt.plot(x, sdy)
plt.title("Standard deviation")

plt.show()

