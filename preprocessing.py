import numpy as np
import csv
def get_data(datafile, ftcount):
	data = list(csv.reader(open(datafile), delimiter=' '))
	train = np.zeros((len(data),ftcount))
	# converts "15:1" to row[15] = 1
	idx = 0
	for row in data:
	    current = train[idx]
	    current[ftcount-1] = int(row[0])
	    for elt in row[1:]:
	        split = elt.split(':')
	        if (len(split) == 2):
	            ftnum = int(split[0])
	            if (ftnum <= len(current)-1):
	                current[ftnum] = int(split[1])
	    idx = idx + 1
	return train

def mask_unused_features(data):
	mask = np.zeros_like(data)
	mask[:] = 1
	empty_columns = data.any(axis=0)
	mask[:, empty_columns] = 0
	data_masked = np.ma.masked_array(data, mask)
	return data_masked

def remove_features(data):
	#removes over-represented features
	return data.T[data.sum(axis=0) < 240].T
def remove_data(data):
	#removes examples with too much features
	return data[data.sum(axis=1) < 110]