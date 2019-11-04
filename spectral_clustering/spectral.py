import numpy as np
from numpy import linalg as LA
from kmeans import kmeans

def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    N = W.shape[0]

    D = np.diag(np.sum(W, axis=1))

    L = D-W

    w, v = LA.eig(L)

    w_idx = np.argsort(w)

    #print(w.T)
    #print(w,v)
    #print(v[:,w_idx[1]])

    map_W  = W @ (v[:,w_idx[0:k]].real)

    return kmeans(map_W.reshape(N, k), k)

    # end answer
