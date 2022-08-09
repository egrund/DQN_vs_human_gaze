import perturbation_for_sarfa as pert
from my_reader_class import Reader
from sample_trajectory import preprocess_image
import dqn

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

FRAME_SKIPS = 4
I = 120 # index of frame (1 to data.get_number_frames())
MODE = 'black' #'blurred' # 'black', 'white', 'random'
SIGMA = 5 # size of perturbation

data = Reader() #file_dir = , images_dir = ) # add path of txt file and 
model = dqn.model(9)
model(tf.random.uniform(shape=(1,84,84,4)),training = False)
model.load_weights('asterix_test/run2/model') # add path

original_image = tf.convert_to_tensor(data.get_image(I))
image = preprocess_image(tf.convert_to_tensor(original_image),84,84)
saliency, perturbed_image = pert.calc_sarfa_saliency_for_image(image,model,mode=MODE,sigma=SIGMA,frame_skips=FRAME_SKIPS)
# saliency, perturbed_image = pert.my_perturbance_map(image,model,mode=MODE,sigma=SIGMA,frame_skips=FRAME_SKIPS)

# plots
fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

axs[0,0].set_title('Original Image')
axs[0,0].imshow(original_image, cmap = 'gray')
axs[0,0].axis('off')

axs[0,1].set_title('Perturbed Image example (also preprocessed already)')
axs[0,1].imshow(pert.image_to_size(perturbed_image), cmap = 'gray')
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