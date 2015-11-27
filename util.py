import numpy as np
import matplotlib.pyplot as plt
import plot_faces
from scipy.io import loadmat
from random import shuffle

def LoadData(filename, labeled=True, unlabeled=True):

  assert ((labeled or unlabeled) and not (labeled and unlabeled)), "Only one dataset must be loaded."
  """Loads data for labeled images
  @ data['tr_identity']: an anonymous identifier unique to a given individual. This is not the image Id.
  @ data['tr_labels']: the labels for each image. array of 2925 ints 1-7
  @ data['tr_images']: the images given by pixel matrices (32 pixels by 32 pixels by 2925 images)
                        a 3D array of shape [32][32][2925]
  """
  if labeled:

    data = loadmat(filename)
    target_train = data['tr_labels']
    inputs_train = data['tr_images']
    x,y,z = inputs_train.shape
    inputs_train = inputs_train.reshape(x*y, z)

    #train_size = int(0.7 * z)
    #training_set = (inputs_train.T)[:train_size]
    #train_set_labels = target_train[:train_size]
    #validation_set = (inputs_train.T)[train_size:]
    #validation_set_labels = target_train[train_size:]

    #plot_faces.plot_digits(training_set[:9])
    #plot_faces.plot_digits(validation_set[:9])

  return target_train, inputs_train.T
  #return training_set, train_set_labels, validation_set, validation_set_labels


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

if __name__ == '__main__':
  LoadData('labeled_images.mat', True, False)



