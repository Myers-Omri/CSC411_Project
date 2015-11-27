__author__ = 'omrim'

from util import LoadData
import sklearn.linear_model

if __name__ == '__main__':


    training_set, train_set_labels, validation_set, validation_set_labels = LoadData('labeled_images.mat', True, False)


    logistic_regression_solver = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.01, C=1.0, fit_intercept=True,
                                            intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg',
                                            max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)



    x = training_set.reshape(2047, 1024)
    y = train_set_labels.reshape(2047,)
    fl = logistic_regression_solver.fit(x, y)
    xv = validation_set(2925-2047, 1024)
    yv = validation_set_labels(2925-2047,)
    accuracy = fl.score(xv, yv)
    print accuracy


