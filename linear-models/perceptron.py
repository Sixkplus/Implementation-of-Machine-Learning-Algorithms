import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE
    
    # begin answer
    
    # [1, X[0], ..., X[P-1]]
    X_new = np.vstack((np.ones((1, X.shape[1])), X))

    for i in range(N):
        cur_X, cur_Y = X_new[:,i], y[0,i]
        while( np.sign( np.matmul(w.T, cur_X) ) != cur_Y ):
            w += (cur_Y * cur_X).reshape((P + 1, 1))
            iters += 1


    # end answer
    
    return w, iters