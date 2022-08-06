from my_reader_class import Reader
import heatmap_comparison as compare 
import perturbation_for_sarfa as pert
from sample_trajectory import preprocess_image
from dqn import DQN
from sarfa_saliency import computeSaliencyUsingSarfa


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

data = Reader()

I1 = 1
I2 = 2
# compare two human gaze heatmaps
heatmap1 = data.create_gaze_heatmap(I1)
heatmap2 = data.create_gaze_heatmap(I2)
auc, map1, map2 = compare.heatmap_comparison_using_AUC(heatmap1, heatmap2)
print("AUC gaze to gaze: ", auc)
fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 8),label="maps: gaze to gaze")
axs[0,0].set_title('Map1')
axs[0,0].imshow(map1)
axs[0,0].axis('off')
axs[0,1].set_title('Map2')
axs[0,1].imshow(map2)
axs[0,1].axis('off') 
plt.show()

# compare human gaze heatmap with sarfa saliency heatmap
# check perturbation_sarfa_example.py
I = 500
FRAME_SKIPS = 4
MODE = 'blurred'
heatmap = data.create_gaze_heatmap(I)
model = DQN(9)
model(tf.random.uniform(shape=(1,84,84,4)),training = False)
#model.load_weights() # add path
original_image = tf.convert_to_tensor(data.get_image(I))
image = preprocess_image(original_image,84,84)
observation = tf.repeat(image,FRAME_SKIPS,axis=-1) # model gets several times the same image
q_vals = tf.squeeze(model(tf.expand_dims(observation,axis=0),training = False),axis=0)
action = tf.argmax(q_vals).numpy()
masks = pert.create_masks(image) # one mask for every pixel
saliency = np.zeros(shape=(len(masks)))
for i,mask in enumerate(masks):
    p_image = tf.convert_to_tensor(pert.perturb_image(image.numpy(),mask, mode=MODE))
    observation = tf.repeat(p_image,FRAME_SKIPS,axis=-1) # model gets several times the same image
    p_q_vals = tf.squeeze(model(tf.expand_dims(observation,axis=0),training = False),axis = 0)
    sal,_,_,_,_,_ = computeSaliencyUsingSarfa(action,pert.array_to_dict(q_vals.numpy()),pert.array_to_dict(p_q_vals.numpy()))
    saliency[i] = sal
saliency = pert.image_to_size(tf.reshape(saliency,shape=image.shape))

auc, map1, map2 = compare.heatmap_comparison_using_AUC(heatmap, saliency)

print("AUC gaze to dqn: ", auc)
fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 8),label="maps: gaze to dqn")
axs[0,0].set_title('Map1')
axs[0,0].imshow(map1)
axs[0,0].axis('off')
axs[0,1].set_title('Map2')
axs[0,1].imshow(map2)
axs[0,1].axis('off') 
plt.show()
