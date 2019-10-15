import numpy as np

def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))

    # YOUR CODE HERE
    # begin answer

    # [1, X[0], ..., X[P-1]]
    X_new = np.vstack((np.ones((1, X.shape[1])), X))
    lr = 1e-3
    thr = 1e-4

    y = (y+1)/2
    # Calculate the gradient
    exp_w_x = np.exp(np.matmul(w.T, X_new))
    d_w = 1/(1+exp_w_x) * (-X_new*y + X_new*(1-y)*exp_w_x)
    d_w = np.sum(d_w, axis=1, keepdims=True)

    while abs((d_w * lr).max()) > thr:
        
        w -= d_w * lr
        #print("dw:",d_w.T, "w:", w.T)
        #print("w:",w)
        exp_w_x = np.exp(np.matmul(w.T, X_new))
        d_w = 1/(1+exp_w_x) * (-X_new*y + X_new*(1-y)*exp_w_x)
        d_w = np.sum(d_w, axis=1, keepdims=True)


    # end answer
    
    return w
