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

I = 700 # index of frame (1 to data.get_number_frames())
SIGMA = 2.8 # size of perturbation

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/") 
model = AgentModel(9)
model.load_weights('asterix_test/run8/model') # add path

original_image = tf.convert_to_tensor(data.get_image(I))
image = preprocess_image(tf.convert_to_tensor(original_image),84,84)

mask = np.zeros(shape=image.shape)
mask[32,30] = 1
mask = ndi.gaussian_filter(mask, sigma = SIGMA)
mask = mask / mask[32,30]

p_black = tf.zeros(shape=image.shape)
p_blurred = ndi.gaussian_filter(image, sigma=5)
p_white = tf.fill(image.shape,tf.reduce_max(image))
p_random = tf.random.uniform(shape=image.shape, minval = np.min(image), maxval = np.max(image))

pi_black = pert.perturb_image(image,mask = mask,mode='image',perturbation=p_black)
pi_blurred = pert.perturb_image(image,mask = mask,mode='image',perturbation=p_blurred)
pi_white = pert.perturb_image(image,mask = mask,mode='image',perturbation=p_white)
pi_random = pert.perturb_image(image,mask = mask,mode='image',perturbation=p_black)

# plots
fig, axs = plt.subplots(nrows=2, ncols=3, squeeze=False, figsize=(8, 8))

axs[0,0].set_title('Preprocessed Image')
axs[0,0].imshow(image/np.max(image))
axs[0,0].axis('off')

axs[0,1].set_title('Mask')
axs[0,1].imshow(mask)
axs[0,1].axis('off')  

axs[0,2].set_title('Black')
axs[0,2].imshow(pi_black/np.max(pi_black))
axs[0,2].axis('off')  

axs[1,0].set_title('White')
axs[1,0].imshow(pi_white/np.max(pi_white))
axs[1,0].axis('off')  

axs[1,1].set_title('Blurred')
axs[1,1].imshow( pi_blurred/np.max(pi_blurred))
axs[1,1].axis('off')

axs[1,2].set_title('Random')
axs[1,2].imshow(pi_random/np.max(pi_random))
axs[1,2].axis('off')

plt.tight_layout()
plt.show()
