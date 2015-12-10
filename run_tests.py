__author__ = 'omrim'
from run_classifier import *
from util import *

if __name__ == '__main__':
    training_set, train_set_labels, ids = LoadData('labeled_images.mat', True, False)

    public_test = LoadData('public_test_images.mat', False, False)
    # hidden_test_images = LoadData('hidden_test_images.mat', False, False)


    run_my_votin(training_set, train_set_labels, public_test, None, train=True)
    # run_my_votin(training_set, train_set_labels, hidden_test_images, None, train=False)