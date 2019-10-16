import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE
    N_test, P = x.shape
    N, _ = x_train.shape


    x_test = x.reshape([N_test, 1, P])

    # ||x  - x_train ||^2

    # (N_test, N_train, P)
    x_diff = x_test - x_train

    # (N_test, N_train)
    x_dist = np.sum((x_diff*x_diff), axis=2)

    indices = np.argsort(x_dist, axis=1)

    nearest_k_indices = indices[:,0:k]

    vote_results = scipy.stats.mode(y_train[nearest_k_indices],axis=1)

    print(vote_results[1].shape)

    y = vote_results[0]
    '''
    # for loop version
    y = np.zeros(N_test)
    for i in range(N_test):
        cur_x = x[i,:]
        cur_dist = cur_x - x_train
    '''

    # begin answer
    # end answer

    return y
