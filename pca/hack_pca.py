import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pca import PCA

def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)

    # YOUR CODE HERE
    # begin answer

    img = img_r[:,:,0]*0.299 + img_r[:,:,1]*0.587 + img_r[:,:,2]*0.114
    #img /= 255

    data = []
    h, w = img.shape

    for i in range(h):
        for j in range(w):
            if(img[i,j] >= 10 and img[i,j] < 200):
                data.append([i-h/2,j-w/2])
    
    # [N, 2]
    data = np.array(data)
    #data[:,0] -= np.sum(data[:,0])
    #data[:,1] -= np.sum(data[:,1])

    eigvalue, eigvector = PCA(data)

    print(eigvalue)
    print(eigvector)

    img = Image.fromarray(img)
    #print(np.arctan(eigvector[0,1]/eigvector[0,0]))
    angle = np.arccos(eigvector[0,0]/np.sqrt(eigvector[0,0]**2 + eigvector[0,1]**2))*180/np.pi
    #if(eigvector[0,0] > 0 and eigvector[1,0] < 0):
    #    angle = -angle
    img = img.rotate(angle - 90)

    return img

    # end answer