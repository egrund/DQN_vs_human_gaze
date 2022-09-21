# blurring comparison
import perturbation_for_sarfa as pert
from my_reader_class import Reader

from scipy import ndimage as ndi 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

FRAME_SKIPS = 4
I = 20 # index of frame (1 to data.get_number_frames()) # for other index have to create the images
SIGMA = 2.8 # size of perturbation

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45/") 
#data = Reader(file_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/") 
image = tf.convert_to_tensor(data.get_image(I),dtype='uint8')
image = pert.preprocess_image(image,84,84)

#masks = pert.create_masks(image,sigma=SIGMA,step=2) # wieso dauert das so lang?
mask = np.zeros(shape=image.shape)
mask[32,30] = 1
mask = ndi.gaussian_filter(mask, sigma = SIGMA)
mask = mask / mask[32,30]

perturbation = tf.zeros(shape=image.shape)
perturbed_image = pert.perturb_image(image,mask = mask,mode='image',perturbation=perturbation)
#perturbed_image = tf.multiply(image,(1-mask)) # we removed part, but because perturbation is 0, is the same as perturbed

# plots
fig, axs = plt.subplots(nrows=1, ncols=5, squeeze=False, figsize=(8, 8))

axs[0,0].set_title("Image")
axs[0,0].imshow(image/tf.reduce_max(image),cmap="gray")
axs[0,0].axis('off')

axs[0,1].set_title("1 - Mask")
axs[0,1].imshow(1- mask,cmap="gray")
axs[0,1].axis('off')

axs[0,2].set_title("Perturbation") # decision for this
axs[0,2].imshow(perturbation,cmap="gray")
axs[0,2].axis('off')

axs[0,3].set_title("Mask") # not good
axs[0,3].imshow(mask,cmap="gray")
axs[0,3].axis('off')

axs[0,4].set_title("Perturbed Image") # not good
axs[0,4].imshow(perturbed_image/tf.reduce_max(perturbed_image),cmap="gray")
axs[0,4].axis('off')

plt.tight_layout()
plt.show()
