import perturbation_for_sarfa as pert
from my_reader_class import Reader
from sample_trajectory import preprocess_image
from dqn import DQN
from sarfa_saliency import computeSaliencyUsingSarfa

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

FRAME_SKIPS = 4
I = 500 # index of frame (1 to data.get_number_frames())
MODE = 'blurred'
SIGMA = 8 # size of perturbation

data = Reader() #file_dir = , images_dir = ) # add path of txt file and 
model = DQN(9)
model(tf.random.uniform(shape=(1,84,84,4)),training = False)
#model.load_weights() # add path

original_image = tf.convert_to_tensor(data.get_image(I))
image = preprocess_image(tf.convert_to_tensor(original_image),84,84)
observation = tf.repeat(image,FRAME_SKIPS,axis=-1) # model gets several times the same image

q_vals = tf.squeeze(model(tf.expand_dims(observation,axis=0),training = False),axis=0)
action = tf.argmax(q_vals).numpy()

# do perturbation based saliency 

masks = pert.create_masks(image,sigma=SIGMA) # one mask for every pixel
saliency = np.zeros(shape=(len(masks)))
p_image_plot = None

for i,mask in enumerate(masks):
    p_image = tf.convert_to_tensor(pert.perturb_image(image.numpy(),mask, mode=MODE))
    if(i==3570): # middel pixel 84 * 42 + 42
        p_image_plot = p_image
    observation = tf.repeat(p_image,FRAME_SKIPS,axis=-1) # model gets several times the same image

    p_q_vals = tf.squeeze(model(tf.expand_dims(observation,axis=0),training = False),axis = 0)

    sal,_,_,_,_,_ = computeSaliencyUsingSarfa(action,pert.array_to_dict(q_vals.numpy()),pert.array_to_dict(p_q_vals.numpy()))
    saliency[i] = sal

saliency = tf.reshape(saliency,shape=image.shape)

# plots

fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

axs[0,0].set_title('Original Image')
axs[0,0].imshow(original_image, cmap = 'gray')
axs[0,0].axis('off')

axs[0,1].set_title('Perturbed Image example (also preprocessed already)')
axs[0,1].imshow(pert.image_to_size(p_image_plot), cmap = 'gray')
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