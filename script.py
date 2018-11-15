
import numpy as np
import preprocessing
import postprocessing
# import vis
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm

ftcount = 531

datafile = 'Dataset/dataset.train'

train = preprocessing.get_data(datafile, ftcount)

trainm = preprocessing.mask_unused_features(train)
#trainm = trainm[trainm.sum(axis=1) < 110]
N = 100
Npoints = 50
# evaluation_tot = np.zeros((Npoints,4))
# evaluation_full = np.zeros((Npoints, N));

for n in range(N):
	print("{0:.2%}".format(float(n)/N))
	evaluation = np.zeros((Npoints,4))
	train_size = 0.8;
	# for i,train_size in enumerate(np.linspace(0.1, 1, Npoints, endpoint=False)):
	for i, Cval in enumerate(np.linspace(0.01, 2., Npoints, endpoint=False)):
		train, test = train_test_split(trainm, train_size=train_size, test_size=1-train_size)#, random_state=666)
		clf = LogisticRegression(solver='liblinear', penalty="l2", C=Cval).fit(train[:,:-1], train[:,-1])
		#clf = svm.SVC(gamma='auto', C=1).fit(train[:,:-1], train[:,-1])
		prediction = clf.predict(test[:,:-1])
		false, missed, correct = postprocessing.evaluate(prediction, test[:,-1])
		evaluation[i,:] = Cval, false, missed, correct
		# evaluation_full[i,n] = correct;
	# evaluation_tot = evaluation_tot + evaluation / N
# np.savetxt("evaluation_dump_LogReg_T95_l1_N{}.dat".format(N), evaluation_tot)
# plt.plot(evaluation_tot[:,0], evaluation_tot[:,3], label="correct")
# plt.plot(evaluation_tot[:,0], evaluation_tot[:,2], label="false positive")
# plt.plot(evaluation_tot[:,0], evaluation_tot[:,1], label="missed positive")
# plt.xlabel("Training proportion")
# plt.xlabel("C l2 penalty value")
# plt.ylabel("Evaluation proportions")
# plt.legend()
# plt.title("{} runs for LogReg".format(N))
means = []
stds = []
for i in range(Npoints):
	means.append(np.mean(evaluation_full[i,:]));
	stds.append(np.std(evaluation_full[i,:]));
plt.errorbar(np.linspace(0.1, 1, Npoints), means, fmt='.', yerr=stds)
_, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(np.linspace(0.1, 1, Npoints, endpoint=False),means)
ax1.set_title("Means")
ax2.plot(np.linspace(0.1, 1, Npoints, endpoint=False),stds)
ax2.set_title("Stds")
plt.show()
#clf2 = svm.SVC(gamma=0.001, C=100).fit(trainm_train[:,:-1], trainm_train[:,-1])
#print(clf.score(trainm_test[:,:-1], trainm_test[:,-1]))
#print(clf2.score(trainm_test[:,:-1], trainm_test[:,-1]))
#virus_proportion = (np.sum(trainm[:,-1])/trainm.shape[0])
#print(virus_proportion)
#clf.predict(trainm_test[:,:-1])