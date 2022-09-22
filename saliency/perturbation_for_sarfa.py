from sarfa_saliency import computeSaliencyUsingSarfa

import numpy as np
import tensorflow as tf
from scipy import ndimage as ndi 
import tensorflow as tf
from imageio.v2 import imread
import os

@tf.function
def preprocess_image(image, imgx = 84, imgy = 84):
    """ preprocesses the image in the same way, as we did while training """
    image = tf.cast(image,tf.float32)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image,size=(imgx,imgy))
    return image

def create_masks(image,sigma=5,step_hor = 2,step_ver = 2):
    """creates an image with a mask around point (x,y) with radius sigma 
    
    Arguments: 
        image (numpy array) : an image of shape (x,y,1) 
        sigma (int>0) : controls the size of the mask
        step_hor (int>0) : how often to create a mask horizontally (for every pixel = 1, for every second = 2)
        step_ver (int>0) : how often to create a mask vertically (for every pixel = 1, for every second = 2)
    """
    
    masks = []
    x_sal = 0
    y_sal = 0
    for x in range(0,image.shape[0],step_ver):
        x_sal += 1
        for y in range(0,image.shape[1],step_hor):
            y_sal += 1
            mask = np.zeros(shape=image.shape)
            mask[x,y] = 1
            mask = ndi.gaussian_filter(mask, sigma = sigma)
            mask = mask / mask[x,y]
            masks.append(mask)
    return masks, x_sal, int(y_sal/x_sal)

def perturb_image(image,mask,mode='blurred',perturbation = None):
    """ perturbes an image with the burred(black,image,random) version at the mask area 
    
    Arguments: 
        image (Tensor)
        mask: image marks are to change
        mode (String): how to change the masked area
        perturbation (image): if mode='image', this will be the perturbation
    """

    shape = image.shape
    if(mode != 'image'): # blurred default
        perturbation = ndi.gaussian_filter(image, sigma=5) # blurred
        if(mode == 'black'):
            perturbation = tf.zeros(shape=shape)
        elif(mode == 'random'):
            perturbation = tf.random.uniform(shape=image.shape, minval = np.min(image), maxval = np.max(image))
        elif(mode == 'white'):
            perturbation = tf.fill(shape,tf.reduce_max(image))

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

def calc_sarfa_saliency(images, model, mode = 'blurred', masks = None, perturbation = None):
    """ calculates the saliency for an whole image 
    
    Arguments: 
        image (List): the perprocessed images to stack
        model: the NN to predict the values
        mode (String): mode of perturbation see perturb_image
        frame_skips (int): how many frames the model gets at once
    """

    # observation = tf.repeat(image,frame_skips,axis=-1) # model gets several times the same image
    # stack the 4 images
    observation = tf.concat(images,axis=-1)
    q_vals = tf.squeeze(model(tf.expand_dims(observation,axis=0),training = False),axis=0)
    action = tf.argmax(q_vals).numpy()
    if masks == None:
        masks = create_masks(images[0],sigma=2.8) # one mask for every pixel
    masks, x_sal, y_sal = masks
    saliency = np.zeros(shape=(len(masks),1)) # in case image is colourful

    for i,mask in enumerate(masks):
        # perturb all images at the same spot
        p_images = [tf.convert_to_tensor(perturb_image(image.numpy(),mask, mode,perturbation)) for image in images]
        observation = tf.concat(p_images,axis=-1)

        # calcualte the new q-values given the perturbed images
        p_q_vals = tf.squeeze(model(tf.expand_dims(observation,axis=0),training = False),axis = 0)
        sal = computeSaliencyUsingSarfa(action,array_to_dict(q_vals),array_to_dict(p_q_vals))
        saliency[i] = sal

    # resize to input image of this function (was preprocessed)
    saliency = image_to_size(np.reshape(saliency, newshape=(x_sal,y_sal,1)),images[0].shape[0],images[0].shape[1])

    return saliency

def my_perturbance_map(images,model,mode='blurred',masks = None, perturbation = None):
    """creates a binary map with pixels being blurred around changing the action having a value of 1, pixels not changing the action having a value of zero """

    observation = tf.concat(images,axis=-1)
    q_vals = tf.squeeze(model(tf.expand_dims(observation,axis=0),training = False),axis=0)
    action = tf.argmax(q_vals).numpy()
    if masks == None:
        masks = create_masks(images[0],sigma=5) # one mask for every pixel
    saliency = np.zeros(shape=(images[0].shape[0],images[0].shape[1],1)) # in case image is colourful

    for mask,x,y in masks:
        p_images = [tf.convert_to_tensor(perturb_image(image.numpy(),mask, mode,perturbation)) for image in images]

        observation = tf.concat(p_images,axis=-1)
        p_q_vals = tf.squeeze(model(tf.expand_dims(observation,axis=0),training = False),axis = 0)
        p_action = tf.argmax(p_q_vals).numpy()
        if(p_action != action):
            saliency[x][y] = 1

    return saliency

def load_saliency(start,last = None,path="",step=1):
    """ loads in every 'step'th saliency start or start to last from given path 
    returns saliency if last is None, else a List of saliencies """

    FRAME_SKIPS = 4

    if last == None:
        saliency = np.array(imread(os.path.join(path,str(start)+"-"+str(start+FRAME_SKIPS-1) + ".png")))
        return saliency / np.max(saliency.astype(np.float32)) # make sure in [0-1]
    
    saliencies = [ imread(os.path.join(path,str(i)+"-"+str(i+FRAME_SKIPS-1) + ".png")) for i in range(start,last +1,step)]
    saliencies = [ saliency /np.max(saliency.astype(np.float32)) for saliency in saliencies]
    return saliencies
