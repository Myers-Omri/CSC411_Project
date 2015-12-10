__author__ = 'omrim'

from util import LoadData, barplot_bagging, barplot_preprocess,gabor_filter
import sklearn.linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.mixture import GMM
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn import svm
import cPickle as pickle
from util import standard_data, fix_pixels
import matplotlib.pyplot as plt
from neural_nets import load_net_and_check_errorate, net_class
#if pre true will make the pre process in the function else work with the data as is
def knn(training_inputs, training_labels, valid_inputs, valid_label, pre=True):
    knn_class = KNeighborsClassifier(weights='distance', n_neighbors=13)
    if pre:
        standard_train_inputs = fix_pixels(training_inputs)
        standard_valid_inputs  = fix_pixels(validation_set)
    else:
        standard_train_inputs = training_inputs
        standard_valid_inputs = valid_inputs

    fitted_knn = knn_class.fit(standard_train_inputs, np.ravel(training_labels))
    res_f = open('trained_lr.dump', 'w')
    pickle.dump(fitted_knn,res_f )
    res_f.close()

    accuracy = knn_class.score(standard_valid_inputs, np.ravel(valid_label))

    print "Accuracy for knn is:{}".format(accuracy)
    return accuracy

#if pre true will make the pre process in the function else work with the data as is
def logistic_regression(training_inputs, training_labels, valid_inputs, valid_label, pre=True):
    logistic_regression_solver = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.008, C=1.2, fit_intercept=True,
                                                                         intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg',
                                                                         max_iter=150, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)

    if pre:
        standard_train_inputs = standard_data(training_inputs)
        standard_valid_inputs = standard_data(valid_inputs)
    else:
        standard_train_inputs = training_inputs
        standard_valid_inputs = valid_inputs

    fl = logistic_regression_solver.fit(standard_train_inputs, training_labels.ravel())
    res_f = open('trained_lr.dump', 'w')
    pickle.dump(fl,res_f )
    res_f.close()

    accuracy = fl.score(standard_valid_inputs, np.ravel(valid_label))
    print 'the accuracy for logistic regression is:',accuracy
    return accuracy
    #print "Accuracy for logistic regression is:{}".format(score_arr)



#if pre true will make the pre process in the function else work with the data as is
def  run_svm(training_set,  train_set_labels, validation_set, validation_set_labels, pre=True):
    from sklearn import decomposition

    # training_set, validation_set, train_set_labels, validation_set_labels = cross_validation.train_test_split(
    #                 all_data_in, all_data_labels, test_size = 0.3, random_state=1, stratify=ids)

    if pre:
        standard_train_inputs = fix_pixels(training_set)
        standard_valid_inputs = fix_pixels(validation_set)
    else:
        standard_train_inputs = training_set
        standard_valid_inputs = validation_set


    clf = svm.SVC(kernel='rbf', C=50, shrinking = False,decision_function_shape='ovr', tol=0.001, max_iter=-1)

    clf.fit(standard_train_inputs, train_set_labels.ravel())

    accuracy = clf.score(standard_valid_inputs, validation_set_labels.ravel())

    res_f = open('trained_svm.dump', 'w')
    pickle.dump(clf,res_f )
    res_f.close()
    print "the new best acc is:" , accuracy, 'the prams are g={}, c={}'.format(0,50)
    return accuracy


def get_prior_dist(training_labels):
    stats = [0,0,0,0,0,0,0]
    for i,l in enumerate(training_labels):
        stats[l-1] += 1
    llen = len(training_labels)
    stts = [float(s) / llen for s in stats]
    print(stts)

def run_voting(training_set, train_set_labels, validation_set, validation_set_labels):
    from sklearn.ensemble import VotingClassifier
    standard_train_inputs = standard_data(training_set)
    standard_valid_inputs = standard_data(validation_set)
    kknn_class = KNeighborsClassifier(weights='uniform', n_neighbors=5)

    logistic_regression_solver = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.01, C=1.0, fit_intercept=True,
                                                                         intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg',
                                                                         max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=2)
    svm_class = svm.SVC(decision_function_shape='ovo', tol=0.001)
    eclf1 = VotingClassifier(estimators=[('knn', kknn_class), ('lr', logistic_regression_solver), ('svm', svm_class)], voting='hard')
    eclf1.fit(standard_train_inputs,train_set_labels.ravel())

    accuracy = eclf1.score(standard_valid_inputs,validation_set_labels.ravel())
    print accuracy

def run_bagging(training_set, train_set_labels,  clsf,validation_set=None, validation_set_labels=None , facc=False):
    from sklearn.ensemble import BaggingClassifier

    bgc = BaggingClassifier(base_estimator=clsf, n_estimators=11, max_samples=1.0, max_features=1.0, bootstrap=True,
                            bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None,
                            verbose=0)
    # standard_train_inputs = standard_data(training_set)
    # standard_valid_inputs = standard_data(validation_set)
    fbgc = bgc.fit(training_set,train_set_labels.ravel())
    if facc:
        acc = fbgc.score(validation_set,validation_set_labels.ravel())
        print(acc)
        return acc
    if facc:
        return fbgc



def get_acc(preds, labels, multy=False ):
    corrects = 0
    errs = []

    if multy:
        pred = [(np.argmax(ar, axis=0)+1) for ar in preds]
    else:
        pred = preds
    for p,l in zip(pred,labels):
        corrects += int(p==l[0])
        errs.append((p,l[0]))
    return (float(corrects) / len(labels)) , errs

def create_csv(labels,fname):
    import csv

    myfile = open(fname, 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(['Id', 'Prediction'])
    i=0
    for l in labels:
        wr.writerow([i+1, l])
        i+=1
    while i<1253:
        wr.writerow([i+1, 0])
        i+=1
    myfile.close()



def run_my_votin(training_set, train_set_labels, validation_set=None, validation_set_labels=None, train=True):
    from sklearn.ensemble import VotingClassifier
    from pybrain.datasets import ClassificationDataSet


    standard_valid_inputs = standard_data(validation_set)
    fixed_valid = fix_pixels(validation_set)
    equalize_and_standard_validation= standard_data(fixed_valid)
    if train:
        standard_train_inputs = standard_data(training_set)
        fixed_train_set = fix_pixels(training_set)
        equalize_and_standard = standard_data(fixed_train_set)

        kknn_class = KNeighborsClassifier(weights='distance', n_neighbors=11)
        # kknn_class.fit(standard_train_inputs, train_set_labels.ravel())
        logistic_regression_solver = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.01, C=1.0, fit_intercept=True,
                                                                             intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg',
                                                                             max_iter=200, multi_class='ovr', verbose=0, warm_start=False, n_jobs=2)
        svm_class = svm.SVC(kernel='rbf', C=50, shrinking = False,decision_function_shape='ovr', tol=0.001, max_iter=-1)


        bg1 = run_bagging(fixed_train_set, train_set_labels, kknn_class, None, None, False)
        res_f = open('bg1knn.dump', 'w')
        pickle.dump(bg1,res_f )
        res_f.close()
        print "Knn done"
        bg2 = run_bagging(standard_train_inputs, train_set_labels, logistic_regression_solver, None, None, False)
        res_f = open('bg2lr.dump', 'w')
        pickle.dump(bg2,res_f )
        res_f.close()
        print "done bg LR"
        bg3 = run_bagging(equalize_and_standard, train_set_labels ,svm_class,  None, None, False)
        res_f = open('bg3svm.dump', 'w')
        pickle.dump(bg3,res_f )
        res_f.close()
        print "done bg svm"
        net_clf = net_class(standard_train_inputs,train_set_labels, None, None, False)
        print "nets done"
    else:

        res_1 = open('bg1knn.dump', 'r')
        bg1 = pickle.load(res_1)
        res_1.close()
        print "knn done"
        res_2 = open('bg2lr.dump', 'r')
        bg2 = pickle.load(res_2)
        res_2.close()
        print "LR done"
        res_3 = open('bg3svm.dump', 'r')
        bg3 = pickle.load(res_3)
        res_3.close()
        print "svm done"

        res_4 = open('bestNet.dump', 'r')
        net_clf = pickle.load(res_4)
        res_4.close()
        print "net done"

    # vds = ClassificationDataSet(1024, 8, nb_classes=8)
    # lX = standard_data(standard_valid_inputs)
    # for vd, vt in zip(lX, Y):
    #     vtarr = [int(i==vt) for i in range(0,8)]
    #     vds.addSample(vd, vtarr)
    # net_predictions = net_clf.testOnClassData(dataset=vds)

    # eclf1 = VotingClassifier(estimators=[('knn', kknn_class), ('lr', bg2), ('svm', bg3)], voting='soft')
    # eclf1.fit(standard_train_inputs,train_set_labels.ravel())
    # print "done fit votings"
    # voting_probs = eclf1.predict_proba(standard_valid_inputs)

    preds_arr = []
    pred_weights = [0.05, 0.4,0.45]
    net_weight = 0.1



    preds_arr.append(bg1.predict_proba(fixed_valid))
    preds_arr.append(bg2.predict_proba(standard_valid_inputs))
    preds_arr.append(bg3.predict_proba(equalize_and_standard_validation))

    net_preds =[]
    for in_data in standard_valid_inputs:
        net_preds.append(net_clf.activate(in_data))

    # preds_arr.append(net_preds)
    fin_pred = []
    for i in range(len(standard_valid_inputs)):
        tmp_np = np.zeros(7)
        for w ,pp in zip(pred_weights, preds_arr):
            tmp_np += pp[i] * w
        tmp_np += net_preds[i] * net_weight

        fin_pred.append(tmp_np)




    # for i,pr in enumerate(fin_pred):
    #     pr += voting_probs[i]*0.75

    fin_labels = [(np.argmax(ar, axis=0)+1) for ar in fin_pred]
    create_csv(fin_labels,'res_csv.csv')
    # print(fin_labels)
    if not validation_set_labels == None:
        fin_acc, err = get_acc(fin_labels, validation_set_labels)
        print 'The final accuracy after bagging and votig is :', fin_acc
    # print "and thats all the errors"
    # print [(x,y) for x,y in err if x==7]

    # prob_predictions = []
    # for bgc in [bg1, bg2, bg3]:
    #     prob_predictions.append(bgc.predict_proba(validation_set.ravel()))
    # prob_predictions.append(bg4.activateOnDataset(validation_set.ravel()))

    # final_pred = []
    # for ip, p in enumerate(train_set_labels):
    #     for i,v in enumerate(prob_predictions):
    #         final_pred

def run_public_test_on(class_name):

    if class_name == 'knn':
        res_1 = open('bg1knn.dump', 'r')
        clf = pickle.load(res_1)
        res_1.close()
        print "knn done"
    elif class_name == 'lr':
        res_2 = open('bg2lr.dump', 'r')
        clf = pickle.load(res_2)
        res_2.close()
        print "LR done"
    elif class_name == 'svm':
        res_3 = open('bg3svm.dump', 'r')
        clf = pickle.load(res_3)
        res_3.close()
        print "svm done"
    elif class_name == 'nn':
        res_4 = open('bestNet.dump', 'r')
        clf = pickle.load(res_4)
        res_4.close()
        print "net done"
    validation_set = LoadData('public_test_images.mat', False, False)
    fixed_valid = fix_pixels(validation_set)
    fin_pred = clf.predict_proba(fixed_valid)
    fin_labels = [(np.argmax(ar, axis=0)+1) for ar in fin_pred]
    create_csv(fin_labels,'res_csv.csv')



def make_data_for_barplot():
    accuracys = []
    training_set, train_set_labels, validation_set, validation_set_labels = LoadData('labeled_images.mat', True, True)
    # training_set, train_set_labels, idst = LoadData('labeled_images.mat', True, False)

    kknn_class = KNeighborsClassifier(weights='distance', n_neighbors=5)
    logistic_regression_solver = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.001, C=1.2, fit_intercept=True,
                                                                             intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg',
                                                                             max_iter=200, multi_class='ovr', verbose=0, warm_start=False, n_jobs=2)
    svm_class = svm.SVC(kernel='rbf', C=50, shrinking = False,decision_function_shape='ovr', tol=0.001, max_iter=-1)

    standard_train_inputs = standard_data(training_set)
    standard_valid_inputs = standard_data(validation_set)

    fixed_train_set = fix_pixels(training_set)
    fixed_valid = fix_pixels(validation_set)


    accuracys.append(knn(training_sett, train_set_labels, validation_set, validation_set_labels))
    print"knn"
    accuracys.append(logistic_regression(training_sett, train_set_labels, validation_set, validation_set_labels))
    print"logistic_regression"
    accuracys.append(run_svm(training_sett, train_set_labels, validation_set, validation_set_labels))
    print"run_svm"

    accuracys.append( run_bagging(fixed_train_set, train_set_labels, kknn_class,fixed_valid, validation_set_labels, True))
    print" knn B"
    accuracys.append( run_bagging(standard_train_inputs, train_set_labels, logistic_regression_solver,standard_valid_inputs, validation_set_labels, True))
    print"logistic_regression  B"
    accuracys.append( run_bagging(fixed_train_set, train_set_labels, svm_class,fixed_valid, validation_set_labels, True))
    print"run_svm  B"

    create_csv(accuracys,'barplot_bagg_accuracy.csv')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    barplot_bagging(ax,accuracys)

    return accuracys

def make_data_for_prepro():
    accuracys = []
    training_sett, train_set_labelts, validation_set, validation_set_labels = LoadData('labeled_images.mat', True, True)
    # training_set, train_set_labels, idst = LoadData('labeled_images.mat', True, False)
    # kknn_class = KNeighborsClassifier(weights='distance', n_neighbors=5)
    # logistic_regression_solver = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.001, C=1.2, fit_intercept=True,
    #                                                                          intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg',
    #                                                                          max_iter=200, multi_class='ovr', verbose=0, warm_start=False, n_jobs=2)
    # svm_class = svm.SVC(kernel='rbf', C=50, shrinking = False,decision_function_shape='ovr', tol=0.001, max_iter=-1)

    standard_train_inputs = standard_data(training_sett)
    standard_valid_inputs = standard_data(validation_set)

    fixed_train_set = fix_pixels(training_sett)
    fixed_valid = fix_pixels(validation_set)

    # garbored_train_set = gabor_filter(training_sett)
    # garbored_valid_set = gabor_filter(validation_set)

    data_list = [(training_sett,validation_set), (standard_train_inputs, standard_valid_inputs),
                 (fixed_train_set,fixed_valid)]#,(garbored_train_set,garbored_valid_set)]
    for (t,v) in data_list:

        # accuracys.append(knn(t, train_set_labelts, v, validation_set_labels, False))
        # accuracys.append(logistic_regression(t,train_set_labelts , v, validation_set_labels, False))
        # accuracys.append(run_svm(t, train_set_labelts, v, validation_set_labels, False))
        net_clf = net_class(t, train_set_labelts, v, validation_set_labels, False)
        net_preds =[]
        for in_data in v:
            net_preds.append(net_clf.activate(in_data))
        accuracys.append(get_acc(net_preds,validation_set_labels, True))
        print"done iter"

    create_csv(accuracys,'barplot_pre_accuracy.csv')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    barplot_preprocess(ax,accuracys)



if __name__ == '__main__':
    training_sett, train_set_labelts, validation_set, validation_set_labels = LoadData('labeled_images.mat', True, True)
    #training_inputs, valid_inputs, training_labels, valid_label = cross_validation.train_test_split(train_inputs, train_targets, test_size=0.3, rlogistic_regression(training_set, train_set_labels, validation_set, validation_set_labels)
    # knn(training_set, train_set_labels, validation_set, validation_set_labels)
    #logistic_regression(training_set, train_set_labels, validation_set, validation_set_labels)
    # MoG(training_set, train_set_labels, validation_set, validation_set_labels)
    #adaBoost(training_set, train_set_labels, validation_set, validation_set_labels)
    # inp, labels, ids = LoadData('labeled_images.mat', True, False)
    # run_svm(inp, labels, ids)


    # fixed_train_set = fix_pixels(training_sett)
    # fixed_valid = fix_pixels(validation_set)
    # standard_train_inputs = standard_data(training_sett)
    # standard_valid_inputs = standard_data(validation_set)
    # kknn_class = KNeighborsClassifier(weights='distance', n_neighbors=5)
    # clsf = svm.SVC(kernel='rbf', C=50, shrinking = False,decision_function_shape='ovr', tol=0.001, max_iter=-1)
    # # run_bagging(fixed_train_set, train_set_labelts, clsf, fixed_valid, validation_set_labels, True)
    # logistic_regression_solver = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.001, C=1.2, fit_intercept=True,
    #                                                                          intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg',
    #                                                                          max_iter=200, multi_class='ovr', verbose=0, warm_start=False, n_jobs=2)

    # run_bagging(standard_train_inputs, train_set_labelts, logistic_regression_solver, standard_valid_inputs, validation_set_labels, True)
    # logistic_regression(standard_train_inputs, train_set_labelts,  standard_valid_inputs, validation_set_labels, pre=True)
    #run_voting(training_set, train_set_labels, validation_set, validation_set_labels)

    # create_csv(range(11,21))

    training_set, train_set_labels, idst = LoadData('labeled_images.mat', True, False)
    # validation_set= LoadData('public_test_images.mat', False, False)
    run_my_votin(training_set, train_set_labels,validation_set, None, False)
    # run_public_test_on('svm')

    # run_svm(training_sett,  validation_set, train_set_labelts,validation_set_labels)
    # make_data_for_barplot()
    # make_data_for_prepro()





