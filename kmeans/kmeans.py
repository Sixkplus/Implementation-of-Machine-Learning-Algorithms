import numpy as np


def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    # YOUR CODE HERE

    N, P = x.shape

    # begin answer

    random_permutation = np.random.permutation(N)
    center_indices = random_permutation[0:k]
    
    x_origin = x
    x = np.reshape(x, [N, 1, P])

    flag = True

    iter_ctrs = []

    centers =  x_origin[center_indices]
    cur = k
    for i in range(k):
        while centers[i] in centers[:i]:
            centers[i] = x_origin[cur]
            cur += 1
    iter_ctrs.append(centers)

    while(flag):
        # [N, k, P]
        x_diff = x - centers

        # (N, k)
        x_dist = np.sum((x_diff*x_diff), axis=2)

        # Assign the cluster for each point 
        indices = np.argmin(x_dist, axis=1)

        new_centers = np.zeros((k, P))

        flag = False
        for cur_k in range(k):
            new_centers[cur_k] = np.mean( x_origin[indices == cur_k, :], axis=0 )
            if (new_centers[cur_k] == centers[cur_k]).min() == 0:
                flag = True
        #print(centers)
        centers = new_centers
        iter_ctrs.append(new_centers)

    ctrs = centers
    idx = indices
    iter_ctrs = np.array(iter_ctrs)


    # end answer

    return idx, ctrs, iter_ctrs
