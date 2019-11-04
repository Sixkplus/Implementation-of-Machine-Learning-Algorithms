import numpy as np

def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    '''

    # YOUR CODE HERE
    # begin answer

    N, P = X.shape

    x_test = np.reshape(X, [N, 1, P])

    # ||X_test  - X ||^2

    # (N, N, P)
    x_diff = x_test - X

    # (N, N)
    W_tot = np.exp(-np.sum((x_diff*x_diff), axis=2)/(2*threshold*threshold))

    indices = np.argsort(-W_tot, axis=1)

    # include the self point w_{ii}
    nearest_k_indices = indices[:,0:k+1]

    W = np.zeros([N,N])

    #W[nearest_k_indices,:] = W_tot[nearest_k_indices,:]

    for i in range(N):
        W[i,nearest_k_indices[i]] = W_tot[i,nearest_k_indices[i]]
        W[i,i] = 0

    return W

    # end answer
