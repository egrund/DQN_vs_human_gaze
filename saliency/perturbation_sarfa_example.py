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
MODE = 'black' #'blurred' # 'black', 'white', 'random'
SIGMA = 5 # size of perturbation

data = Reader(file_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/") 
model = AgentModel(9)
model.load_weights('asterix_test/run8/model') # add path

original_image = tf.convert_to_tensor(data.get_image(I))
image = preprocess_image(tf.convert_to_tensor(original_image),84,84)
masks1 = pert.create_masks(image,sigma=SIGMA,step=2)
masks2 = pert.create_masks(image,sigma=SIGMA,step=1)

#perturbed_image = pert.perturb_image(image,mask = masks[400][0],mode=MODE)
start = time.time()
saliency1 = pert.calc_sarfa_saliency_for_image(image,model,mode=MODE,masks=masks1,frame_skips=FRAME_SKIPS)
saliency = ndi.gaussian_filter(saliency1, sigma=0.5)
saliency2 = pert.calc_sarfa_saliency_for_image(image,model,mode=MODE,masks=masks2,frame_skips=FRAME_SKIPS)
#bi_saliency = pert.my_perturbance_map(image,model,mode=MODE,masks=masks,frame_skips=FRAME_SKIPS)
end = time.time()
print("Time needed: ",end - start)

# plots
fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

axs[0,0].set_title('every Pixel')
axs[0,0].imshow(pert.image_to_size(saliency2))
axs[0,0].axis('off')

axs[0,1].set_title('1')
axs[0,1].imshow(pert.image_to_size( ndi.gaussian_filter(saliency1, sigma=1)))
#axs[0,1].imshow(original_image,cmap='gray',alpha=0.5)
axs[0,1].axis('off') 

axs[1,0].set_title('0.75')
axs[1,0].imshow(pert.image_to_size( ndi.gaussian_filter(saliency1, sigma=0.75)))
axs[1,0].axis('off')  

axs[1,1].set_title('0.5')
axs[1,1].imshow(pert.image_to_size( ndi.gaussian_filter(saliency1, sigma=0.5)))
#axs[1,1].imshow(original_image, cmap = 'gray', alpha=0.5)
axs[1,1].axis('off')

plt.tight_layout()
plt.show()
