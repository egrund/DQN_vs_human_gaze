import integrated_gradients as ig 
from my_reader_class import Reader
from sample_trajectory import preprocess_image
from model import AgentModel

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

FRAME_SKIPS = 4
MODE = 'Black'
M_STEPS = 10
I = 200 # index of frame (1 to data.get_number_frames())

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/")
model = AgentModel(9)
model.load_weights('asterix_test/run8/model') # add path

original_image = data.get_image(I)
image = preprocess_image(tf.convert_to_tensor(original_image),84,84)
observation = tf.repeat(image,FRAME_SKIPS,axis=-1) # model gets several times the same image

# use action model decides as right or use human decision as right one
# action = tf.argmax(tf.squeeze(model(tf.expand_dims(observation,axis=0),training = False),axis = 0)).numpy()
action = data.get_action(I)

# create integrated gradients 
baseline = ig.create_baseline(observation,mode=MODE)

# calcualte integrated_gradients
integrated_gradients = ig.integrated_gradients(model,baseline,observation,action,m_steps=M_STEPS)

# make integrated_gradients and baseline back to shape of one image
baseline_image = ig.create_baseline(image,mode=MODE)
ig_image = tf.expand_dims(np.mean(integrated_gradients,axis=-1),axis=-1)

# https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/integrated_gradients/integrated_gradients.ipynb

#convergence_check(model, integrated_gradients, baseline, observation, action)

attribution_mask = tf.reduce_sum(tf.math.abs(ig_image), axis=-1)

# Visualization
fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

axs[0,0].set_title('Baseline Image')
axs[0,0].imshow(baseline_image)
axs[0,0].axis('off')

axs[0,1].set_title('Original Image')
axs[0,1].imshow(original_image, cmap = 'gray')
axs[0,1].axis('off') 

axs[1,0].set_title('IG Attribution Mask')
axs[1,0].imshow(attribution_mask, cmap=plt.cm.inferno)
axs[1,0].axis('off')  

axs[1,1].set_title('Original + IG Attribution Mask Overlay')
axs[1,1].imshow(attribution_mask, cmap=plt.cm.inferno)
axs[1,1].imshow(image, cmap = 'gray', alpha=0.2)
axs[1,1].axis('off')

plt.tight_layout()
plt.show()