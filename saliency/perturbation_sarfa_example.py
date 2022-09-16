# blurring comparison
import perturbation_for_sarfa as pert
from my_reader_class import Reader
from sample_trajectory import preprocess_image
from model import AgentModel

from scipy import ndimage as ndi 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from imageio.v2 import imread

FRAME_SKIPS = 4
I = 700 # index of frame (1 to data.get_number_frames()) # for other index have to create the images
MODE = 'image' #'blurred' # 'black', 'white', 'random'
SIGMA = 2.8 # size of perturbation

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/") 
#data = Reader(file_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/") 

model = AgentModel(9)
model.load_weights('asterix_test/run8/model') # add path

original_images = [ tf.convert_to_tensor(data.get_image(i)) for i in range(I,I+4)]
#images_dif = [ preprocess_image(tf.convert_to_tensor(original_image),84,84) for original_image in original_images]
images_same = [preprocess_image(tf.convert_to_tensor(original_images[0]),84,84)] *4
masks1 = pert.create_masks(images_same[0],sigma=SIGMA,step=2)
#masks2 = pert.create_masks(images[0],sigma=SIGMA,step=1)
perturbation = tf.zeros(shape=images_same[0].shape)

#perturbed_image = pert.perturb_image(images_same,mask = masks1[200],mode=MODE,perturbation = perturbation)
#saliency_dif = pert.calc_sarfa_saliency_for_image(images_dif,model,mode=MODE,masks=masks1,perturbation = perturbation)
#saliency_dif = ndi.gaussian_filter(ndi.gaussian_filter(saliency_dif, sigma=0.8),sigma=0.8)
saliency_same = pert.calc_sarfa_saliency_for_image(images_same,model,mode=MODE,masks=masks1,perturbation = perturbation)
saliency_same = ndi.gaussian_filter(ndi.gaussian_filter(saliency_same, sigma=0.8),sigma=0.8)

# calculate
#saliencies = pert.calc_sarfa_saliency_for_each_image(images_dif,model,mode='image',masks=masks1,perturbation = perturbation)
#saliencies = [ndi.gaussian_filter(ndi.gaussian_filter(saliency, sigma=0.8),sigma=0.8) for saliency in saliencies]

# load
saliencies = [ imread("D:/Documents/Gaze_Data_Project/saliency_database_test/" + str(I) + "_" + str(j) + ".png") for j in range(0,FRAME_SKIPS,1)]
saliency700_4 = imread("D:/Documents/Gaze_Data_Project/saliency_database_test/" + str(I) + ".png")

# add all saliencies up
added_sal = np.sum(saliencies,axis=0) 

# check only where all big value
sal_max = tf.reduce_max(saliencies)
saliency_only_all_indexes = np.sum(np.where((saliencies/sal_max) > 0.2, 1, 0),axis=0) # make zero where is smaller than 0.2, add all up to one map
saliency_only_all = np.where(saliency_only_all_indexes == 4,added_sal,0)

# plots
fig, axs = plt.subplots(nrows=3, ncols=4, squeeze=False, figsize=(8, 8))

for i in range(4):
    axs[0,i].set_title('Image ' + str(i+1))
    axs[0,i].imshow(original_images[i])
    axs[0,i].axis('off')

for i in range(4):
    axs[1,i].set_title('Saliency ' + str(i+1))
    axs[1,i].imshow(saliencies[i]/tf.reduce_max(saliencies))
    axs[1,i].imshow(original_images[i],alpha=0.5)
    axs[1,i].axis('off')

axs[2,0].set_title("I+4(dif),Added up") # not good
axs[2,0].imshow(added_sal/tf.reduce_max(added_sal))
axs[2,0].imshow(original_images[3],alpha=0.5)
axs[2,0].axis('off')

axs[2,1].set_title("Saliency same Image ")
axs[2,1].imshow(pert.image_to_size(saliency_same))
axs[2,1].imshow(original_images[0],alpha=0.5)
axs[2,1].axis('off')

axs[2,2].set_title("I+4 (dif), one map") # decision for this, because maybe sometimes the images are different
axs[2,2].imshow(saliency700_4)
axs[2,2].imshow(original_images[0],alpha=0.5)
axs[2,2].axis('off')

axs[2,3].set_title("I+4 where all big") # not good
axs[2,3].imshow(saliency_only_all)
axs[2,3].imshow(original_images[0],alpha=0.5)
axs[2,3].axis('off')

plt.tight_layout()
plt.show()
