__author__ = 'omrim'

from util import LoadData
import sklearn.linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
import numpy as np


def standard_data(inputs):
    mean = np.mean(inputs, axis=0)
    std = np.std(inputs, axis=0)
    return (inputs - mean) / std

# def knn(training_inputs, valid_inputs, training_labels, valid_label):
#     knn = KNeighborsClassifier(weights = 'uniform', n_neighbors = 5)
#
#     #training_inputs, valid_inputs, training_labels, valid_label = cross_validation.train_test_split(train_inputs, train_targets, test_size=0.3, random_state=0)
#     #Train knn
#     standard_train_inputs = standard_data(training_inputs)
#     knn.fit(standard_train_inputs, training_labels.ravel())
#      #Get the accuracy of the model
#     standard_valid_inputs = standard_data(valid_inputs)
#     accuracy = knn.score(standard_valid_inputs, valid_label.ravel())
#     print "Accuracy for knn is:{}".format(accuracy)

def knn(training_inputs, training_labels):
    knn_class = KNeighborsClassifier(weights='uniform', n_neighbors=5)

    #training_inputs, valid_inputs, training_labels, valid_label = cross_validation.train_test_split(train_inputs, train_targets, test_size=0.3, random_state=0)
    #Train knn
    standard_train_inputs = standard_data(training_inputs)
    fitted_knn = knn_class.fit(standard_train_inputs, training_labels.ravel())

    score_arr = sklearn.cross_validation.cross_val_score(fitted_knn, standard_train_inputs, training_labels.ravel(), scoring=None,
                                             cv=cross_validation.KFold(training_labels.size, 4), n_jobs=1, verbose=0,
                                             fit_params=None, pre_dispatch='2*n_jobs')


    #accuracy = knn_class.score(standard_valid_inputs, valid_label.ravel())
    print "Accuracy for knn is:{}".format(score_arr)

def logistic_regression(training_labels, training_inputs):



    logistic_regression_solver = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.1, C=1.0, fit_intercept=True,
                                                                         intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg',
                                                                         max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=2)
    standard_train_inputs = standard_data(training_inputs)
    fl = logistic_regression_solver.fit(standard_train_inputs, training_labels.ravel())

    score_arr = sklearn.cross_validation.cross_val_score(fl, standard_train_inputs, training_labels.ravel(), scoring=None,
                                             cv=cross_validation.KFold(training_labels.size, 4), n_jobs=1, verbose=0,
                                             fit_params=None, pre_dispatch='2*n_jobs')


  #  accuracy = fl.score(standard_valid_inputs, valid_label.ravel())
    print "Accuracy for logistic regression is:{}".format(score_arr)

def multi_naive_bayes(training_inputs, valid_inputs, training_labels, valid_label):
    mnb = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)



def neural_nets():
    print ' '

def get_prior_dist():
    stats = [0,0,0,0,0,0,0]
    for i,l in enumerate(training_labels):
        stats[l-1] += 1
    llen = len(training_labels)
    stts = [float(s) / llen for s in stats]
    print(stts)

if __name__ == '__main__':
    train_targets, train_inputs = LoadData('labeled_images.mat', True, False)
    training_inputs, valid_inputs, training_labels, valid_label = cross_validation.train_test_split(train_inputs, train_targets, test_size=0.3, random_state=0)



    #logistic_regression(train_targets, train_inputs)
    knn(training_inputs, training_labels)
    #multi_naive_bayes(training_inputs, valid_inputs, training_labels, valid_label)
    # neural_nets()



