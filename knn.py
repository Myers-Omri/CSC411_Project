import numpy as np
from util import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from scipy.io import loadmat


def knn():
	#train_inputs, train_label, valid_inputs, valid_label = LoadData('labeled_images.mat', True, False)
	train_targets, train_inputs = LoadData('labeled_images.mat', True, False)
	knn = KNeighborsClassifier(weights = 'uniform', n_neighbors = 5)

	training_inputs, valid_inputs, training_labels, valid_label = cross_validation.train_test_split(train_inputs, train_targets, test_size=0.3, random_state=0)
	#Train knn

	knn.fit(training_inputs, training_labels.ravel())

	#Get prediction on valid inputs
	#prediction = knn.predict(valid_inputs)
	#print(prediction)
	#print(valid_label.ravel())

	#Get the accuracy of the model
	accuracy = knn.score(valid_inputs, valid_label.ravel())
	print(accuracy)

if __name__ == '__main__':
	knn()