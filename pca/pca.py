import numpy as np
from numpy import linalg as LA

def PCA(data):
    '''
    PCA	Principal Component Analysis

    Input:
      data      - Data numpy array. Each row vector of fea is a data point.
    Output:
      eigvector - Each column is an embedding function, for a new
                  data point (row vector) x,  y = x*eigvector
                  will be the embedding result of x.
      eigvalue  - The sorted eigvalue of PCA eigen-problem.
    '''

    # YOUR CODE HERE
    # Hint: you may need to normalize the data before applying PCA
    # begin answer
    
    # print(data.shape)
    N, P = data.shape
    # [N, P]
    X = data

    # [P,]
    x_mean = np.sum(X, axis=0)

    # [P, P]
    S = 1/N * (X - x_mean).T @ (X - x_mean)

    w, v = LA.eig(S)

    w_idx = np.argsort(-w)

    eigvalue = w[w_idx]

    eigvector = (v[:, w_idx]).T

    return eigvalue, eigvector

    # end answer