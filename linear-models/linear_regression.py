import numpy as np

def linear_regression(X, y):
    '''
    LINEAR_REGRESSION Linear Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))

    # YOUR CODE HERE
    # begin answer

    # [1, X[0], ..., X[P-1]]
    X_new = np.vstack((np.ones((1, X.shape[1])), X))

    w  = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_new, X_new.T)), X_new), y.T)


    # end answer
    return w
