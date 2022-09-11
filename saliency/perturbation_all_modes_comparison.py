# visualizes a comparison of all perturbation modes for sarfa perturbation saliency calculation

import perturbation_for_sarfa as pert
from my_reader_class import Reader
from sample_trajectory import preprocess_image
from model import AgentModel

from scipy import ndimage as ndi 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

FRAME_SKIPS = 4
I = 700 # index of frame (1 to data.get_number_frames())
MODE = ['black' ,'blurred' , 'white', 'random' ]
SIGMA = 5 # size of perturbation

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/") 
model = AgentModel(9)
model.load_weights('asterix_test/run8/model') # add path

original_image = tf.convert_to_tensor(data.get_image(I))
image = preprocess_image(tf.convert_to_tensor(original_image),84,84)
masks = pert.create_masks(image,sigma=SIGMA,step=2)
p_black = tf.zeros(shape=image.shape) # save 10 seconds per image by only creating this one
p_blurred = ndi.gaussian_filter(image, sigma=5)
p_white = tf.ones(shape=image.shape)
p_random = tf.random.uniform(shape=image.shape, minval = 0.0, maxval = 1.0)

saliency_black = ndi.gaussian_filter(pert.calc_sarfa_saliency_for_image(image,model,mode='image',masks=masks,perturbation=p_black,frame_skips=FRAME_SKIPS), sigma=0.5)
saliency_blurred = ndi.gaussian_filter(pert.calc_sarfa_saliency_for_image(image,model,mode='image',masks=masks,perturbation=p_blurred,frame_skips=FRAME_SKIPS), sigma=0.5)
saliency_white = ndi.gaussian_filter(pert.calc_sarfa_saliency_for_image(image,model,mode='image',masks=masks,perturbation=p_white,frame_skips=FRAME_SKIPS), sigma=0.5)
saliency_random = ndi.gaussian_filter(pert.calc_sarfa_saliency_for_image(image,model,mode='image',masks=masks,perturbation=p_random,frame_skips=FRAME_SKIPS), sigma=0.5)

saliency_total = saliency_black + saliency_blurred + saliency_white + saliency_random
# plots
fig, axs = plt.subplots(nrows=3, ncols=2, squeeze=False, figsize=(8, 8))

axs[0,0].set_title('original image')
axs[0,0].imshow(original_image)
axs[0,0].axis('off')

axs[0,1].set_title('total')
axs[0,1].imshow(pert.image_to_size(saliency_total))
#axs[0,1].imshow(original_image,cmap='gray',alpha=0.5)
axs[0,1].axis('off') 

axs[1,0].set_title('black')
axs[1,0].imshow(pert.image_to_size( saliency_black))
axs[1,0].axis('off')  

axs[1,1].set_title('blurred')
axs[1,1].imshow(pert.image_to_size( saliency_blurred))
#axs[1,1].imshow(original_image, cmap = 'gray', alpha=0.5)
axs[1,1].axis('off')

axs[2,0].set_title('white')
axs[2,0].imshow(pert.image_to_size( saliency_white))
axs[2,0].axis('off')  

axs[2,1].set_title('random')
axs[2,1].imshow(pert.image_to_size( saliency_random))
#axs[2,1].imshow(original_image, cmap = 'gray', alpha=0.5)
axs[2,1].axis('off')

plt.tight_layout()
plt.show()
