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
SIGMA = 2.8 # size of perturbation

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/") 
model = AgentModel(9)
model.load_weights('asterix_test/run8/model') # add path

original_images = [ tf.convert_to_tensor(data.get_image(i)) for i in range(I,I+FRAME_SKIPS)]
images = [ preprocess_image(tf.convert_to_tensor(original_image),84,84) for original_image in original_images]
masks = pert.create_masks(images[0],sigma=SIGMA)
p_black = tf.zeros(shape=images[0].shape)
p_blurred = ndi.gaussian_filter(images[0], sigma=5)
p_white = tf.fill(images[0].shape,tf.reduce_max(images[0]))
p_random = tf.random.uniform(shape=images[0].shape,  minval = np.min(images[0]), maxval = np.max(images[0]))

saliency_black = pert.calc_sarfa_saliency(images,model,mode='image',masks=masks,perturbation=p_black)
saliency_blurred = pert.calc_sarfa_saliency(images,model,mode='image',masks=masks,perturbation=p_blurred)
saliency_white = pert.calc_sarfa_saliency(images,model,mode='image',masks=masks,perturbation=p_white)
saliency_random = pert.calc_sarfa_saliency(images,model,mode='image',masks=masks,perturbation=p_random)

saliency_total = saliency_black + saliency_blurred + saliency_white + saliency_random
# plots
fig, axs = plt.subplots(nrows=2, ncols=3, squeeze=False, figsize=(8, 8))

axs[0,0].set_title('Image 4')
axs[0,0].imshow(original_images[3])
axs[0,0].axis('off')

axs[0,1].set_title('Black')
axs[0,1].imshow(pert.image_to_size( saliency_black),cmap="jet")
axs[0,1].imshow(original_images[3],cmap='gray',alpha=0.8)
axs[0,1].axis('off')  

axs[0,2].set_title('White')
axs[0,2].imshow(pert.image_to_size( saliency_white),cmap="jet")
axs[0,2].imshow(original_images[3],cmap='gray',alpha=0.8)
axs[0,2].axis('off')  

axs[1,0].set_title('Blurred')
axs[1,0].imshow(pert.image_to_size( saliency_blurred),cmap="jet")
axs[1,0].imshow(original_images[3], cmap = 'gray', alpha=0.8)
axs[1,0].axis('off')

axs[1,1].set_title('Random')
axs[1,1].imshow(pert.image_to_size( saliency_random),cmap="jet")
axs[1,1].imshow(original_images[3], cmap = 'gray', alpha=0.8)
axs[1,1].axis('off')

axs[1,2].set_title('All Added')
axs[1,2].imshow(pert.image_to_size(saliency_total),cmap="jet")
axs[1,2].imshow(original_images[3],cmap='gray',alpha=0.8)
axs[1,2].axis('off') 

plt.tight_layout()
plt.show()
