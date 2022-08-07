from sarfa_saliency import computeSaliencyUsingSarfa
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

def image_to_size(image,y=160,x=210):
    """ reshapes the image to a given size 
    
    Arguments: 
        image (Tensor)
        y = horizontal size
        x = vertical size
    """

    image = tf.image.resize(image,size=(x,y))
    return image

def calc_sarfa_saliency_for_image(image, model, mode = 'blurred',sigma = 8, frame_skips = 4):
    """ calculates the saliency for an whole image 
    
    Arguments: 
        image: the perprocessed image
        model: the NN to predict the values
        mode (String): mode of perturbation see perturb_image
        sigma
        frame_skip (int): how many frames the model gets at once
    """

    observation = tf.repeat(image,frame_skips,axis=-1) # model gets several times the same image
    q_vals = tf.squeeze(model(tf.expand_dims(observation,axis=0),training = False),axis=0)
    action = tf.argmax(q_vals).numpy()

    masks = create_masks(image,sigma=sigma) # one mask for every pixel
    saliency = np.zeros(shape=(len(masks)))
    perturbed_example_image = None

    for i,mask in enumerate(masks):
        p_image = tf.convert_to_tensor(perturb_image(image.numpy(),mask, mode=mode))
        if(i==3570): # middel pixel 84 * 42 + 42
            perturbed_example_image = p_image
        observation = tf.repeat(p_image,frame_skips,axis=-1) # model gets several times the same image

        p_q_vals = tf.squeeze(model(tf.expand_dims(observation,axis=0),training = False),axis = 0)
        sal,_,_,_,_,_ = computeSaliencyUsingSarfa(action,array_to_dict(q_vals.numpy()),array_to_dict(p_q_vals.numpy()))
        saliency[i] = sal

    saliency = tf.reshape(saliency,shape=image.shape)

    return saliency, perturbed_example_image

