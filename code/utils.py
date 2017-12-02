# Utilities for the NN exercises in ISTA 421, Introduction to ML

import numpy
from numpy import matlib
import math
import visualize
import matplotlib.pyplot as plt
import gradient

import random

iter = 0

# -------------------------------------------------------------------------
@numpy.vectorize
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

# -------------------------------------------------------------------------

def initialize(hidden_size, visible_size):
    """
    Sample weights uniformly from the interval [-r, r] as described in lecture 23.
    Return 1d array theta (in format as described in Exercise 2)
    :param hidden_size: number of hidden units
    :param visible_size: number of visible units (of input and output layers of autoencoder)
    :return: theta array
    """

    ### YOUR CODE HERE ###
    u_range = - math.sqrt(6 / (visible_size * 2 + 1) )
    # visible to hidden
    W1 = numpy.random.uniform(-u_range, u_range, (visible_size, hidden_size)).flatten()
    # hidden to output
    W2 = numpy.random.uniform(-u_range, u_range, (hidden_size, visible_size)).flatten()
    # Hidden bias
    b1 = numpy.random.uniform(-u_range, u_range, (hidden_size)).flatten()
    # Output bias
    b2 = numpy.random.uniform(-u_range, u_range, (visible_size)).flatten()

    theta = numpy.concatenate((W1, W2, b1, b2))
    print('theta shapes: ', W1.shape, W2.shape, b1.shape, b2.shape)

    assert(theta.shape[0] == 
            visible_size * hidden_size + 
            hidden_size * visible_size +
            hidden_size +
            visible_size)

    print('theta shape ', theta.shape)


    return theta



@numpy.vectorize
def loss(x, y):
    return 0.5 * numpy.linalg.norm(x - y) ** 2

@numpy.vectorize
def hadamard_sq(x):
    return x ** 2

@numpy.vectorize
def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))
# -------------------------------------------------------------------------

def autoencoder_cost_and_grad_nonv(theta, visible_size, hidden_size, lambda_, data):
    # Decompose vector into components
    W1_index = visible_size * hidden_size
    W2_index = W1_index + hidden_size * visible_size
    b1_index = W2_index + hidden_size
    b2_index = b1_index + visible_size
    # hidden_size x visible_size
    W1 = numpy.matrix(theta[0:W1_index]).reshape((hidden_size, visible_size))
    W2 = numpy.matrix(theta[W1_index: W2_index]).reshape((visible_size, hidden_size))
    b1 = numpy.matrix(theta[W2_index: b1_index]).reshape(1, hidden_size)
    b2 = numpy.matrix(theta[b1_index: b2_index]).reshape(1, visible_size)
    assert(b2_index == theta.shape[0])
    m = data.shape[1]


    for i, x in enumerate(data.T):
        x = numpy.matrix(x)
        z2 = (W1 * x.transpose()).transpose() + b1
        #print(W1.shape, x.transpose().shape, b1.shape)
        print('z2', z2.shape)
        a2 = sigmoid(z2)
        z3 = W2 * a2[-1].transpose() + b2
        h = sigmoid(z3)

        cost = 1 / m * numpy.matrix.sum(loss(h, x))
        weight_decay = 0.5 * lambda_ * (hadamard_sq(W1).sum() + hadamard_sq(W2).sum())
        cost += weight_decay


        delta3 = numpy.multiply(-(x - h), sigmoid_deriv(z3))
        delta2 = numpy.multiply(W2.T * delta3, sigmoid_deriv(z2))

        del_W2 = delta3 * (a2.T)
        del_W1 = delta2 * (x.T)


    print(a2.shape, h.shape)


    return a2, h


def autoencoder_cost_and_grad(theta, visible_size, hidden_size, lambda_, data):
    """
    The input theta is a 1-dimensional array because scipy.optimize.minimize expects
    the parameters being optimized to be a 1d array.
    First convert theta from a 1d array to the (W1, W2, b1, b2)
    matrix/vector format, so that this follows the notation convention of the
    lecture notes and tutorial.
    You must compute the:
        cost : scalar representing the overall cost J(theta)
        grad : array representing the corresponding gradient of each element of theta
    """

    ### YOUR CODE HERE ###
    cost, grad = None, None  # implement

    # Decompose vector into components
    W1_index = visible_size * hidden_size
    W2_index = W1_index + hidden_size * visible_size
    b1_index = W2_index + hidden_size
    b2_index = b1_index + visible_size
    # hidden_size x visible_size
    W1 = numpy.matrix(theta[0:W1_index]).reshape((hidden_size, visible_size))
    W2 = numpy.matrix(theta[W1_index: W2_index]).reshape((visible_size, hidden_size))
    b1 = numpy.matrix(theta[W2_index: b1_index]).reshape(1, hidden_size)
    b2 = numpy.matrix(theta[b1_index: b2_index]).reshape(1, visible_size)
    assert(b2_index == theta.shape[0])
    m = data.shape[1]

    #b is hidden_size x 100:w
    z2 = W1 * data + numpy.repeat(b1, m, axis=0).transpose()
    a2 = sigmoid(z2)
    z3 = W2 * a2 + numpy.repeat(b2, m, axis=0).transpose()
    h = sigmoid(z3)

    # For the vectorized way of doing this, we will also need z2 and z3
    # to calculate the gradient
    #a2, h, z2, z3 = autoencoder_feedforward(theta, visible_size, hidden_size, data)
    cost = 1 / m * numpy.matrix.sum(loss(h, data))
    weight_decay = 0.5 * lambda_ * (numpy.square(W1).sum() + numpy.square(W2).sum())
    cost += weight_decay


    delta3 = numpy.multiply(-(data - h), sigmoid_deriv(z3))
    delta2 = numpy.multiply(W2.T * delta3, sigmoid_deriv(z2))

    del_W2 = (delta3 * (a2.T)).flatten()
    del_W1 = (delta2 * (data.T)).flatten()


    # Remove cols added to b1 and b2 by numpy.repeat
    del_b2 = delta3[:,0]
    del_b1 = delta2[:,0]


    grad = numpy.concatenate((
        del_W1.T,
        del_W2.T,
        # b derivs are simply delta
        del_b1,
        del_b2
    ))

    # Verify shapes are good
    assert(grad.shape[0] == theta.shape[0])

    #print('cost ', cost)
    return cost, grad




# -------------------------------------------------------------------------

def autoencoder_cost_and_grad_sparse(theta, visible_size, hidden_size, lambda_, rho_, beta_, data):
    """
    Version of cost and grad that incorporates the hidden layer sparsity constraint
        rho_ : the target sparsity limit for each hidden node activation
        beta_ : controls the weight of the sparsity penalty term relative
                to other loss components

    The input theta is a 1-dimensional array because scipy.optimize.minimize expects
    the parameters being optimized to be a 1d array.
    First convert theta from a 1d array to the (W1, W2, b1, b2)
    matrix/vector format, so that this follows the notation convention of the
    lecture notes and tutorial.
    You must compute the:
        cost : scalar representing the overall cost J(theta)
        grad : array representing the corresponding gradient of each element of theta
    """

    ### YOUR CODE HERE ###
    cost, grad = None, None  # implement

    return cost, grad


# -------------------------------------------------------------------------

def autoencoder_feedforward(theta, visible_size, hidden_size, data):
    """
    Given a definition of an autoencoder (including the size of the hidden
    and visible layers and the theta parameters) and an input data matrix
    (each column is an image patch, with 1 or more columns), compute
    the feedforward activation for the output visible layer for each
    data column, and return an output activation matrix (same format
    as the data matrix: each column is an output activation "image"
    corresponding to the data input).

    Once you have implemented the autoencoder_cost_and_grad() function,
    simply copy the part of the code that computes the feedforward activations
    up to the output visible layer activations and return that activation.
    You do not need to include any of the computation of cost or gradient.
    It is likely that your implementation of feedforward in your
    autoencoder_cost_and_grad() is set up to handle multiple data inputs,
    in which case your only task is to ensure the output_activations matrix
    is in the same corresponding format as the input data matrix, where
    each output column is the activation corresponding to the input column
    of the same column index.

    :param theta: the parameters of the autoencoder, assumed to be in this format:
                  { W1, W2, b1, b2 }
                  W1 = weights of layer 1 (input to hidden)
                  W2 = weights of layer 2 (hidden to output)
                  b1 = layer 1 bias weights (to hidden layer)
                  b2 = layer 2 bias weights (to output layer)
    :param visible_size: number of nodes in the visible layer(s) (input and output)
    :param hidden_size: number of nodes in the hidden layer
    :param data: input data matrix, where each column is an image patch,
                  with one or more columns
    :return: output_activations: an matrix output, where each column is the
                  vector of activations corresponding to the input data columns
    """

    ### YOUR CODE HERE ###
    output_activations = None  # implement

    # Decompose vector into components
    W1_index = visible_size * hidden_size
    W2_index = W1_index + hidden_size * visible_size
    b1_index = W2_index + hidden_size
    b2_index = b1_index + visible_size
    # hidden_size x visible_size
    W1 = numpy.matrix(theta[0:W1_index]).reshape((hidden_size, visible_size))
    W2 = numpy.matrix(theta[W1_index: W2_index]).reshape((visible_size, hidden_size))
    b1 = numpy.matrix(theta[W2_index: b1_index]).reshape(1, hidden_size)
    b2 = numpy.matrix(theta[b1_index: b2_index]).reshape(1, visible_size)
    assert(b2_index == theta.shape[0])
    m = data.shape[1]

    #b is hidden_size x 100:w
    z2 = W1 * data + numpy.repeat(b1, m, axis=0).transpose()
    a2 = sigmoid(z2)
    z3 = W2 * a2 + numpy.repeat(b2, m, axis=0).transpose()
    h = sigmoid(z3)
    print('act shapes', a2.shape, h.shape)

    #return numpy.matrix((a2, h))
    #return a2, h, z2, z3
    return h

    


# -------------------------------------------------------------------------

def save_model(theta, visible_size, hidden_size, filepath, **params):
    """
    Save the model to file.  Used by plot_and_save_results.
    :param theta: the parameters of the autoencoder, assumed to be in this format:
                  { W1, W2, b1, b2 }
                  W1 = weights of layer 1 (input to hidden)
                  W2 = weights of layer 2 (hidden to output)
                  b1 = layer 1 bias weights (to hidden layer)
                  b2 = layer 2 bias weights (to output layer)
    :param visible_size: number of nodes in the visible layer(s) (input and output)
    :param hidden_size: number of nodes in the hidden layer
    :param filepath: path with filename
    :param params: optional parameters that will be saved with the model as a dictionary
    :return:
    """
    numpy.savetxt(filepath + '_theta.csv', theta, delimiter=',')
    with open(filepath + '_params.txt', 'a') as fout:
        params['visible_size'] = visible_size
        params['hidden_size'] = hidden_size
        fout.write('{0}'.format(params))


# -------------------------------------------------------------------------

def plot_and_save_results(theta, visible_size, hidden_size, root_filepath=None,
                          train_patches=None, test_patches=None, show_p=False,
                          **params):
    """
    This is a helper function to streamline saving the results of an autoencoder.
    The visible_size and hidden_size provide the information needed to retrieve
    the autoencoder parameters (w1, w2, b1, b2) from theta.

    This function does the following:
    (1) Saves the parameters theta, visible_size and hidden_size as a text file
        called '<root_filepath>_model.txt'
    (2) Extracts the layer 1 (input-to-hidden) weights and plots them as an image
        and saves the image to file '<root_filepath>_weights.png'
    (3) [optional] train_patches are intended to be a set of patches that were
        used during training of the autoencoder.  Typically these will be the first
        100 patches of the MNIST data set.
        If provided, the patches will be given as input to the autoencoder in
        order to generate output 'decoded' activations that are then plotted as
        patches in an image.  The image is saved to '<root_filepath>_train_decode.png'
    (4) [optional] test_patches are intended to be a different set of patches
        that were *not* used during training.  This permits inspecting how the
        autoencoder does decoding images it was not trained on.  The output activation
        image is generated the same way as in step (3).  The image is saved to
        '<root_filepath>_test_decode.png'

    The root_filepath is used as the base filepath name for all files generated
    by this function.  For example, if you wish to save all of the results
    using the root_filepath='results1', and you have specified the train_patches and
    test_patches, then the following files will be generated:
        results1_model.txt
        results1_weights.png
        results1_train_decode.png
        results1_test_decode.png
    If no root_filepath is provided, then the output will default to:
        model.txt
        weights.png
        train_decode.png
        test_decode.png
    Note that if those files already existed, they will be overwritten.

    :param theta: model parameters
    :param visible_size: number of nodes in autoencoder visible layer
    :param hidden_size: number of nodes in autoencoder hidden layer
    :param root_filepath: base filepath name for files generated by this function
    :param train_patches: matrix of patches (typically the first 100 patches of MNIST)
    :param test_patches: matrix of patches (intended to be patches NOT used in training)
    :param show_p: flag specifying whether to show the plots before exiting
    :param params: optional parameters that will be saved with the model as a dictionary
    :return:
    """

    filepath = 'model'
    if root_filepath:
        filepath = root_filepath + '_' + filepath
    save_model(theta, visible_size, hidden_size, filepath, **params)

    # extract the input to hidden layer weights and visualize all the weights
    # corresponding to each hidden node
    w1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size).T
    filepath = 'weights.png'
    if root_filepath:
        filepath = root_filepath + '_' + filepath
    visualize.plot_images(w1, show_p=False, filepath=filepath)

    if train_patches is not None:
        # Given: train_patches and autoencoder parameters,
        # compute the output activations for each input, and plot the resulting decoded
        # output patches in a grid.
        # You can then manually compare them (visually) to the original input train_patches
        filepath = 'train_decode.png'
        if root_filepath:
            filepath = root_filepath + '_' + filepath
        train_decode = autoencoder_feedforward(theta, visible_size, hidden_size, train_patches)
        visualize.plot_images(train_decode, show_p=False, filepath=filepath)

    if test_patches is not None:
        # Same as for train_patches, but assuming test_patches are patches that were not
        # used for training the autoencoder.
        # Again, you can then manually compare the decoded patches to the test_patches
        # given as input.
        test_decode = autoencoder_feedforward(theta, visible_size, hidden_size, test_patches)
        filepath = 'test_decode.png'
        if root_filepath:
            filepath = root_filepath + '_' + filepath
        visualize.plot_images(test_decode, show_p=False, filepath=filepath)

    if show_p:
        plt.show()


# -------------------------------------------------------------------------

def get_pretty_time_string(t, delta=False):
    """
    Really cheesy kludge for producing semi-human-readable string representation of time
    y = Year, m = Month, d = Day, h = Hour, m (2nd) = minute, s = second, mu = microsecond
    :param t: datetime object
    :param delta: flag indicating whether t is a timedelta object
    :return:
    """
    if delta:
        days = t.days
        hours = t.seconds // 3600
        minutes = (t.seconds // 60) % 60
        seconds = t.seconds - (minutes * 60)
        return 'days={days:02d}, h={hour:02d}, m={minute:02d}, s={second:02d}' \
                .format(days=days, hour=hours, minute=minutes, second=seconds)
    else:
        return 'y={year:04d},m={month:02d},d={day:02d},h={hour:02d},m={minute:02d},s={second:02d},mu={micro:06d}' \
                .format(year=t.year, month=t.month, day=t.day,
                        hour=t.hour, minute=t.minute, second=t.second,
                        micro=t.microsecond)


# -------------------------------------------------------------------------
