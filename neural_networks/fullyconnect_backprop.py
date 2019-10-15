import numpy as np

def fullyconnect_backprop(in_sensitivity, in_, weight):
    '''
    The backpropagation process of fullyconnect
      input parameter:
          in_sensitivity  : the sensitivity from the upper layer, shape: 
                          : [number of images, number of outputs in feedforward]
          in_             : the input in feedforward process, shape: 
                          : [number of images, number of inputs in feedforward]
          weight          : the weight matrix of this layer, shape: 
                          : [number of inputs in feedforward, number of outputs in feedforward]

      output parameter:
          weight_grad     : the gradient of the weights, shape: 
                          : [number of inputs in feedforward, number of outputs in feedforward]
          bias_grad       : the gradient of the bias, shape: 
          ?????????????????????????????????????????????????????????????????????????????????????
                          : [number of outputs in feedforward, 1]
          out_sensitivity : the sensitivity to the lower layer, shape: 
                          : [number of images, number of inputs in feedforward]

    Note : remember to divide by number of images in the calculation of gradients.
    '''

    # TODO
    num_imgs, out_shape = in_sensitivity.shape

    in_shape = weight.shape[0]

    # begin answer
    bias_grad = (np.sum(in_sensitivity, axis=0).T/(num_imgs)).reshape((out_shape,1))
    weight_grad = np.sum(in_sensitivity.reshape((num_imgs, out_shape,1))@in_.reshape((num_imgs, 1, in_shape)) , axis=0).T/(num_imgs)

    out_sensitivity = weight @ in_sensitivity.reshape((num_imgs, out_shape,1))

    out_sensitivity = out_sensitivity.reshape((num_imgs, in_shape))

    #print(weight_grad.shape)
    #print(bias_grad.shape)
    #print(out_sensitivity.shape)

    # end answer

    return weight_grad, bias_grad, out_sensitivity

