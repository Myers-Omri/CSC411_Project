__author__ = 'omrim'

from util import LoadData
import sklearn.linear_model

if __name__ == '__main__':


    training_set, train_set_labels, validation_set, validation_set_labels = LoadData('labeled_images.mat', True, False)


    logistic_regression_solver = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.001, C=1.0, fit_intercept=True,
                                            intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg',
                                            max_iter=200, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)



    fl = logistic_regression_solver.fit(training_set, train_set_labels.ravel())
    accuracy = fl.score(validation_set, validation_set_labels.ravel())
    
    print accuracy


