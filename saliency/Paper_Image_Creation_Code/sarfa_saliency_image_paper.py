# blurring comparison
import perturbation_for_sarfa as pert
from my_reader_class import Reader
from model import AgentModel

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

FRAME_SKIPS = 4
I = 28 # index of frame (1 to data.get_number_frames()) # for other index have to create the images
MODE = 'image' #'blurred' # 'black', 'white', 'random'
SIGMA = 2.8 # size of perturbation

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45/") 
#data = Reader(file_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/") 

model = AgentModel(9)
model.load_weights('asterix_test/best/model') # add path

original_images = [ tf.convert_to_tensor(data.get_image(i)) for i in range(I,I+4)]
images_dif = [ pert.preprocess_image(tf.convert_to_tensor(original_image),84,84) for original_image in original_images]
masks = pert.create_masks(images_dif[0],sigma=SIGMA)
perturbation = tf.zeros(shape=images_dif[0].shape)

saliency = pert.calc_sarfa_saliency(images_dif,model,mode=MODE,masks=masks,perturbation = perturbation)
saliency = pert.image_to_size(saliency/tf.reduce_max(saliency))

# plots
fig, axs = plt.subplots(nrows=1, ncols=5, squeeze=False, figsize=(8, 8))

for i in range(3):
    axs[0,i].set_title('Image ' + str(i+1))
    axs[0,i].imshow(original_images[i])
    axs[0,i].axis('off')

axs[0,3].set_title('Image ' + str(4) + " + Saliency")
axs[0,3].imshow(saliency,cmap="jet")
axs[0,3].imshow(original_images[3],alpha=0.8)
axs[0,3].axis('off')

axs[0,4].set_title('Saliency')
axs[0,4].imshow(saliency,cmap="jet")
axs[0,4].axis('off')

plt.tight_layout()
plt.show()
