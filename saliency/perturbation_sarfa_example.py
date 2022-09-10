import perturbation_for_sarfa as pert
from my_reader_class import Reader
from sample_trajectory import preprocess_image
from model import AgentModel

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

FRAME_SKIPS = 4
I = 700 # index of frame (1 to data.get_number_frames())
MODE = 'black' #'blurred' # 'black', 'white', 'random'
SIGMA = 5 # size of perturbation

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/") #file_dir = , images_dir = ) # add path of txt file and 
model = AgentModel(9)
model.load_weights('asterix_test/run8/model') # add path

original_image = tf.convert_to_tensor(data.get_image(I))
image = preprocess_image(tf.convert_to_tensor(original_image),84,84)
masks = pert.create_masks(image,sigma=SIGMA)

perturbed_image = pert.perturb_image(image,mask = masks[4000][0],mode=MODE)
saliency = pert.calc_sarfa_saliency_for_image(image,model,mode=MODE,masks=masks,frame_skips=FRAME_SKIPS)
# saliency = pert.my_perturbance_map(image,model,mode=MODE,masks=masks,frame_skips=FRAME_SKIPS)

# plots
fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

axs[0,0].set_title('Original Image')
axs[0,0].imshow(original_image, cmap = 'gray')
axs[0,0].axis('off')

axs[0,1].set_title('Preprocessed and perturbed image')
axs[0,1].imshow(pert.image_to_size(perturbed_image/tf.reduce_max(perturbed_image)), cmap = 'gray')
axs[0,1].axis('off') 

axs[1,0].set_title('Saliency')
axs[1,0].imshow(pert.image_to_size(saliency), cmap=plt.cm.inferno)
axs[1,0].axis('off')  

axs[1,1].set_title('Original Image + Saliency')
axs[1,1].imshow(pert.image_to_size(saliency), cmap=plt.cm.inferno)
axs[1,1].imshow(original_image, cmap = 'gray', alpha=0.5)
axs[1,1].axis('off')

plt.tight_layout()
plt.show()