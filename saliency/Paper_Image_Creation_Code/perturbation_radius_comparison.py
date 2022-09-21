# radius of perturbation comparison, for calculating saliency
# decision for 2.8, because smallest value in 2.8 - 3,6 which is the with preprocessing scaled radius for the human gaze heatmap. but seems to have more information than bigger values
import perturbation_for_sarfa as pert
from my_reader_class import Reader
from model import AgentModel

from scipy import ndimage as ndi 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

FRAME_SKIPS = 4
I = 52 # index of frame (1 to data.get_number_frames()) # for other index have to create the images
MODE = 'image' #'blurred' # 'black', 'white', 'random'
SIGMA = [2,2.5,2.8,3,3.5,4] # size of perturbation

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45/") 
#data = Reader(file_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/") 

model = AgentModel(9)
model.load_weights('asterix_test/best/model') # add path

original_images = [ tf.convert_to_tensor(data.get_image(i)) for i in range(I,I+4)]
images_dif = [ pert.preprocess_image(tf.convert_to_tensor(original_image),84,84) for original_image in original_images]
perturbation = tf.zeros(shape=images_dif[0].shape)

# plots
fig, axs = plt.subplots(nrows=2, ncols=3, squeeze=False, figsize=(8, 8))

a = 0
for i,s in enumerate(SIGMA):
    print(s)
    if i == 3:
        a = 1
    masks = pert.create_masks(images_dif[0],sigma=s)
    
    saliency = pert.calc_sarfa_saliency(images_dif,model,mode=MODE,masks=masks,perturbation = perturbation)

    saliency = pert.image_to_size(saliency / tf.reduce_max(saliency))

    axs[a,i-a*3].set_title(s)
    axs[a,i-a*3].imshow(saliency,cmap="jet")
    axs[a,i-a*3].imshow(original_images[3],alpha=0.8) # 0.5 for png, 0.8 for pdf
    axs[a,i-a*3].axis('off')

plt.tight_layout()
plt.show()
