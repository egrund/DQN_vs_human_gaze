import numpy as np
from scipy import ndimage as ndi 
import tensorflow as tf

def create_masks(image,sigma=10):
    """creates an image with a mask around point (x,y) with radius sigma 
    
    Arguments: 
        image (numpy array) : an image of shape (x,y,1) 
        sigma (int>0) : controls the size of the mask
    """
    
    masks = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            mask = np.zeros(shape=image.shape)
            mask[x,y] = 1
            mask = ndi.gaussian_filter(mask, sigma = sigma)
            mask = mask / mask[x,y]
            masks.append(mask)
    return masks

def perturb_image(image,mask,mode='blurred',perturbation = None):
    """ perturbes an image with the burred(black,image,random) version at the mask area 
    
    Arguments: 
        image (Tensor)
        mask: image marks are to change
        mode (String): how to change the masked area
        perturbation (image): if mode='image', this will be the perturbation
    """

    shape = image.shape
    if(mode != 'image'):
        perturbation = ndi.gaussian_filter(image, sigma=5) # blurred
    if(mode == 'black'):
        perturbation = tf.zeros(shape=shape)
    elif(mode == 'random'):
        perturbation = tf.random.uniform(shape=shape, minval = 0.0, maxval = 1.0)

    # Hadamard product
    image1 = tf.multiply(image,(1-mask)) 
    image2 = tf.multiply(perturbation, mask)
    return image1 + image2

def array_to_dict(array):
    """ creates a dictionary with index keys for array for using sarfa-saliency.py"""

    dict = {}
    for i,v in enumerate(array):
        dict[i] = v
    return dict
