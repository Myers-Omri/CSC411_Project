import numpy as np
from util import *
from sklearn.neighbors import KNeighborsClassifier
from scipy.io import loadmat


def knn():
	train_inputs, train_label, valid_inputs, valid_label = LoadData('labeled_images.mat', True, False)

	knn = KNeighborsClassifier(weights = 'uniform', n_neighbors = 5)

	#Train knn
	knn.fit(train_inputs, np.ravel(train_label))

	#Get prediction on valid inputs
	#prediction = knn.predict(valid_inputs)
	#print(prediction)
	#print(valid_label.ravel())

	#Get the accuracy of the model
	accuracy = knn.score(valid_inputs, valid_label.ravel())
	#accuracy = knn.score(train_inputs, train_label.ravel())
	print(accuracy)

if __name__ == '__main__':
	knn()