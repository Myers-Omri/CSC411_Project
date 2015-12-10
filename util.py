import numpy as np
import matplotlib.pyplot as plt
import plot_faces
from scipy.io import loadmat
from random import shuffle
from sklearn import cross_validation
from plot_faces import plot_digits
from skimage.filters import gabor_kernel, gabor_filter
from scipy import ndimage as ndi

import matplotlib.cm as cm
import operator as o
from matplotlib.font_manager import FontProperties





def count_id(input_id):
    list_id = input_id.tolist()
    new_list = [x[0] for x in list_id]

    for i,id in enumerate(new_list):
        if new_list.count(id) == 1:
            new_list[i] = -1

    return np.array(new_list)


#changed load data that we can load splited and unsplited data.
def LoadData(filename, labeled=True, splited=False):

    # assert ((labeled or unlabeled) and not (labeled and unlabeled)), "Only one dataset must be loaded."
    """Loads data for labeled images
    @ data['tr_identity']: an anonymous identifier unique to a given individual. This is not the image Id.
    @ data['tr_labels']: the labels for each image. array of 2925 ints 1-7
    @ data['tr_images']: the images given by pixel matrices (32 pixels by 32 pixels by 2925 images)
                          a 3D array of shape [32][32][2925]
    """
    data = loadmat(filename)
    data2 = loadmat('hidden_test_images.mat')
    if labeled:


        target_train = data['tr_labels']
        inputs_train = data['tr_images']
        input_id = data['tr_identity']
        corrected_list = count_id(input_id)

        x,y,z = inputs_train.shape
        inputs_trainn = (inputs_train.reshape(x*y, z)).T
        if not splited:
            return inputs_trainn, target_train, corrected_list
        if splited:
            training_set, validation_set, train_set_labels, validation_set_labels = cross_validation.train_test_split(
                inputs_trainn, target_train, test_size = 0.3, random_state=1, stratify=corrected_list)

            return training_set, train_set_labels, validation_set, validation_set_labels
    else:
        inputs_public = data['public_test_images']
        inputs_hidden = data2['hidden_test_images']

        x,y,z = inputs_public.shape
        inputs_reshape_public = (inputs_public.reshape(x*y, z)).T

        x,y,z = inputs_hidden.shape
        inputs_reshape_hidden = (inputs_hidden.reshape(x*y, z)).T

        inputs_test = np.vstack((inputs_reshape_public, inputs_reshape_hidden))

        if not splited:
            return inputs_test


def ShowMeans(means, header=''):
    """Show the cluster centers as images."""
    plt.figure(1)
    plt.clf()
    for i in xrange(3):
        plt.subplot(1, means.shape[0], i+1)
        plt.imshow(means[i , :].reshape(32, 32).T, cmap=plt.cm.gray)
    plt.title(header)
    plt.draw()
    raw_input('Press Enter.')

def standard_data(inputs):
    mean = np.mean(inputs, axis=0)
    std = np.std(inputs, axis=0)
    return (inputs - mean) / std

def fix_pixels(inputs):
    from skimage import data, img_as_float
    from skimage import exposure
    new_data = []
    for i in inputs:
        new_i = exposure.equalize_hist(i)
        new_data.append(new_i)
    return new_data




def barplot_bagging(ax, info):
    p = ("K-NN", "Logistic Regression", "SVM")
    dpoints = np.array(    [['Normal Classifier', 'K-NN', info[0]],
                           ['Normal Classifier', 'Logistic Regression', info[1]],
                           ['Normal Classifier', 'SVM', info[2]],

                           ['Using Bagging', 'K-NN', info[3]],
                           ['Using Bagging', 'Logistic Regression', info[4]],
                           ['Using Bagging', 'SVM', info[5]]])
    '''
    Create a barchart for data across different categories with
    multiple conditions for each category.

    @param ax: The plotting axes from matplotlib.
    @param dpoints: The data set as an (n, 3) numpy array
    '''

    # Aggregate the conditions and the categories according to their
    # mean values
    conditions = [(c, np.mean(dpoints[dpoints[:,0] == c][:,2].astype(float)))
                  for c in np.unique(dpoints[:,0])]
    categories = [(c, np.mean(dpoints[dpoints[:,1] == c][:,2].astype(float)))
                  for c in np.unique(dpoints[:,1])]

    # sort the conditions, categories and data so that the bars in
    # the plot will be ordered by category and condition
    conditions = [c[0] for c in sorted(conditions, key=o.itemgetter(1))]
    categories = [c[0] for c in sorted(categories, key=o.itemgetter(1))]

    dpoints = np.array(sorted(dpoints, key=lambda x: categories.index(x[1])))

    # the space between each set of bars
    space = 0.3
    n = len(conditions)
    width = (1 - space) / (len(conditions))

    # Create a set of bars at each position
    for i,cond in enumerate(conditions):
        indeces = range(1, len(categories)+1)
        vals = dpoints[dpoints[:,0] == cond][:,2].astype(np.float)
        pos = [j - (1 - space) / 2. + i * width for j in indeces]
        ax.bar(pos, vals, width=width, label=cond,
               color=cm.Accent(float(i) / n))

    # Set the x-axis tick labels to be equal to the categories
    ax.set_xticks(indeces)
    ax.set_xticklabels(p)

    # Add the axis labels
    ax.set_ylabel("Accuracy")
    #ax.set_ylabel("Time")
    ax.set_xlabel("Classifier")
    fontP = FontProperties()
    fontP.set_size('small')
    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', prop = fontP)

    plt.show()

def barplot_preprocess(ax, info):
    p = ("K-NN", "Logistic Regression", "SVM")
    dpoints = np.array(    [['No-Preprocess', 'K-NN', info[0]],
                           ['No-Preprocess', 'Logistic Regression', info[1]],
                           ['No-Preprocess', 'SVM', info[2]],
                           ['No-Preprocess', 'N-N', info[3]],

                           ['Standardize', 'K-NN', info[4]],
                           ['Standardize', 'Logistic Regression', info[5]],
                           ['Standardize', 'SVM', info[6]],
                           ['Standardize', 'N-N', info[7]],

                           ['Histogram Equalization', 'K-NN', info[8]],
                           ['Histogram Equalization', 'Logistic Regression', info[9]],
                           ['Histogram Equalization', 'SVM', info[10]],
                           ['Histogram Equalization', 'N-N', info[11]],

                           # ['Gabor_filter', 'K-NN', info[12]],
                           # ['Gabor_filter', 'Logistic Regression', info[13]],
                           # ['Gabor_filter', 'SVM', info[14]],
                           # ['Gabor_filter', 'N-N', info[15]],
                            ])
    '''
    Create a barchart for data across different categories with
    multiple conditions for each category.

    @param ax: The plotting axes from matplotlib.
    @param dpoints: The data set as an (n, 3) numpy array
    '''

    # Aggregate the conditions and the categories according to their
    # mean values
    conditions = [(c, np.mean(dpoints[dpoints[:,0] == c][:,2].astype(float)))
                  for c in np.unique(dpoints[:,0])]
    categories = [(c, np.mean(dpoints[dpoints[:,1] == c][:,2].astype(float)))
                  for c in np.unique(dpoints[:,1])]

    # sort the conditions, categories and data so that the bars in
    # the plot will be ordered by category and condition
    conditions = [c[0] for c in sorted(conditions, key=o.itemgetter(1))]
    categories = [c[0] for c in sorted(categories, key=o.itemgetter(1))]

    dpoints = np.array(sorted(dpoints, key=lambda x: categories.index(x[1])))

    # the space between each set of bars
    space = 0.3
    n = len(conditions)
    width = (1 - space) / (len(conditions))

    # Create a set of bars at each position
    for i,cond in enumerate(conditions):
        indeces = range(1, len(categories)+1)
        vals = dpoints[dpoints[:,0] == cond][:,2].astype(np.float)
        pos = [j - (1 - space) / 2. + i * width for j in indeces]
        ax.bar(pos, vals, width=width, label=cond,
               color=cm.Accent(float(i) / n))

    # Set the x-axis tick labels to be equal to the categories
    ax.set_xticks(indeces)
    ax.set_xticklabels(p)
    #plt.setp(plt.xticks()[1], rotation=90)

    # Add the axis labels
    ax.set_ylabel("Accuracy")
    #ax.set_ylabel("Time")
    ax.set_xlabel("Classifier")
    fontP = FontProperties()
    fontP.set_size('small')
    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', prop = fontP)

    plt.show()

if __name__ == '__main__':
    images, labels, ids  = LoadData('labeled_images.mat', True, False)
    filtered_images = gabor_filter_f(images[:20])

    print "originals"
    # plot_digits(images[:9])
    print "new"
    fi = np.matrix(filtered_images[:9])
    # plot_digits(fi)
    print "diff"
    diff_arr = np.concatenate((images[:5], fi[:5]), axis=0 )
    plot_digits(diff_arr)

