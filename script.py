import numpy as np
import preprocessing
import postprocessing
import vis
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm

ftcount = 531

datafile = 'Dataset/dataset.train'

train = preprocessing.get_data(datafile, ftcount)

trainm = preprocessing.mask_unused_features(train)

evaluation_tot = np.zeros((50,4))
N = 1000
for n in range(N):
	print(n)
	evaluation = np.zeros((50,4))
	for i,train_size in enumerate(np.linspace(0.5, 1, 50, endpoint=False)):
		train, test = train_test_split(trainm, train_size=train_size, test_size=1-train_size)#, random_state=666)
		clf = LogisticRegression(solver='liblinear', max_iter=1000).fit(train[:,:-1], train[:,-1])
		#clf = svm.SVC(gamma='auto', C=1).fit(train[:,:-1], train[:,-1])
		prediction = clf.predict(test[:,:-1])
		false, missed, correct = postprocessing.evaluate(prediction, test[:,-1])
		evaluation[i,:] = train_size, false, missed, correct
	evaluation_tot = evaluation_tot + evaluation / N
np.savetxt("evaluation_dump_LogReg_C1_N{}.dat".format(N), evaluation_tot)
plt.plot(evaluation_tot[:,0], evaluation_tot[:,3], label="correct")
plt.plot(evaluation_tot[:,0], evaluation_tot[:,2], label="false positive")
plt.plot(evaluation_tot[:,0], evaluation_tot[:,1], label="missed positive")
plt.xlabel("Training proportion")
plt.ylabel("Evaluation proportions")
plt.legend()
plt.title("{} runs for LogReg".format(N))
plt.show()
#clf2 = svm.SVC(gamma=0.001, C=100).fit(trainm_train[:,:-1], trainm_train[:,-1])
#print(clf.score(trainm_test[:,:-1], trainm_test[:,-1]))
#print(clf2.score(trainm_test[:,:-1], trainm_test[:,-1]))
#virus_proportion = (np.sum(trainm[:,-1])/trainm.shape[0])
#print(virus_proportion)
#clf.predict(trainm_test[:,:-1])