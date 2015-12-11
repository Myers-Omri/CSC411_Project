__author__ = 'omrim'
from run_classifier import *
from util import *
from neural_nets import *

if __name__ == '__main__':

    print '''Hello,
            To train the model and get the predictions for the public+hidden test press 1
            To load the saved model and get the predictions for the public+hidden test press 2'''
    choice = raw_input('')
    training_set, train_set_labels, ids = LoadData('labeled_images.mat', True, False)
    all_test = LoadData('public_test_images.mat', False, False)
    train = False
    if choice == '1':
        train = True
    else:
        train = False

    pred_mat = run_my_votin(training_set, train_set_labels, all_test, None, train)
    pred_file = open('test.dump', 'w')
    pickle.dump(pred_mat, pred_file)
    pred_file.close()

    # for p in pred_mat:
    #     print p

