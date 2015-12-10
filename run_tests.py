__author__ = 'omrim'
from run_classifier import *
from util import *
from neural_nets import *

if __name__ == '__main__':
    training_set, train_set_labels, ids = LoadData('labeled_images.mat', True, False)

    public_test = LoadData('public_test_images.mat', False, False)
    #hidden_test_images = LoadData('hidden_test_images.mat', False, False)

    #standard_train_inputs = standard_data(training_set)
    #net_class(standard_train_inputs,train_set_labels, None, None, False)

    #run_my_votin(training_set, train_set_labels, public_test, None, train=True)
    run_my_votin(training_set, train_set_labels, public_test, None, train=False)