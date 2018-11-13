
import matplotlib.pyplot as plt
import preprocessing

ftcount = 531

datafile = 'Dataset/dataset.train'

train = preprocessing.get_data(datafile, ftcount)

# Count of individuals in each feature
plt.figure(1)
plt.hist(train.sum(axis=0))
plt.title("Count of individual per feature")
plt.ylabel("Number of features")
plt.xlabel("Number of individuals")

# Count of features for each individual
plt.figure(2)
plt.hist(train.sum(axis=1))
plt.title("Count of features per individual")
plt.xlabel("Number of features")
plt.ylabel("Number of individuals")

# Show the overall matrix
plt.figure(3)
plt.imshow(train)

# Show the overall matrix without mask values
trainm = preprocessing.mask_unused_features(train)
plt.figure(4)
plt.imshow(trainm)

plt.show()