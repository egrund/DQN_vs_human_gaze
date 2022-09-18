import integrated_gradients as ig 
from my_reader_class import Reader
from sample_trajectory import preprocess_image
from model import AgentModel
import perturbation_for_sarfa as pert

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

FRAME_SKIPS = 4
MODE_IG = 'Black'
MODE_SARFA = 'image'
M_STEPS = 10
I = 76 # index of frame (1 to data.get_number_frames())
SIGMA = 2.8 # size of perturbation

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/")
model = AgentModel(9)
model.load_weights('asterix_test/run8/model') # add path

original_images = [ tf.convert_to_tensor(data.get_image(i)) for i in range(I,I+4)]
images = [ preprocess_image(tf.convert_to_tensor(original_image),84,84) for original_image in original_images]
observation = tf.concat(images,axis=-1) # for IG

# for SARFA
masks1 = pert.create_masks(images[0],sigma=SIGMA)
perturbation = tf.zeros(shape=images[0].shape) # black
saliency = pert.calc_sarfa_saliency(images,model,mode=MODE_SARFA,masks=masks1,perturbation = perturbation)
saliency= pert.image_to_size(saliency / tf.reduce_max(saliency))

# use action model decides as right or use human decision as right one
action = tf.argmax(tf.squeeze(model(tf.expand_dims(observation,axis=0),training = False),axis = 0)).numpy()
# action = data.get_action(I) # human action

# create integrated gradients 
baseline = ig.create_baseline(observation,mode=MODE_IG)

# calcualte integrated_gradients
integrated_gradients = ig.integrated_gradients(model,baseline,observation,action,m_steps=M_STEPS)

# make integrated_gradients and baseline back to shape of one image
ig_image = tf.expand_dims(np.mean(integrated_gradients,axis=-1),axis=-1)

# https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/integrated_gradients/integrated_gradients.ipynb
attribution_mask = pert.image_to_size(tf.expand_dims(tf.reduce_sum(tf.math.abs(ig_image), axis=-1),axis=-1))

# Visualization
fig, axs = plt.subplots(nrows=1, ncols=4, squeeze=False, figsize=(8, 8))

axs[0,0].set_title('IG')
axs[0,0].imshow(attribution_mask, cmap="jet")
axs[0,0].axis('off')  

axs[0,1].set_title('Image 4 + IG')
axs[0,1].imshow(attribution_mask, cmap="jet")
axs[0,1].imshow(original_images[3], cmap = 'gray', alpha=0.8)
axs[0,1].axis('off')

axs[0,2].set_title("SARFA")
axs[0,2].imshow(saliency,cmap="jet")
axs[0,2].axis('off')

axs[0,3].set_title("Image 4 + SARFA")
axs[0,3].imshow(saliency,cmap="jet")
axs[0,3].imshow(original_images[3], cmap = 'gray', alpha=0.8)
axs[0,3].axis('off')

plt.tight_layout()
plt.show()