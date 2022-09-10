from my_reader_class import Reader
import heatmap_comparison as compare 
import perturbation_for_sarfa as pert
from sample_trajectory import preprocess_image
from model import AgentModel

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/")

# compare human gaze points with sarfa saliency heatmap
# check perturbation_sarfa_example.py

I = 300
MODE = 'black'
gaze_map = data.create_gaze_map(I)
gaze_list = np.array(data.get_gaze(I))
heatmap = data.create_gaze_heatmap(I)
heatmap_other = data.create_gaze_heatmap(I+50)

ig = compare.saliency_information_gain(gaze_map,heatmap)
print("IG gaze to gaze(same index): ", ig) # should be > 0
ig = compare.saliency_information_gain(gaze_map,heatmap_other)
print("IG gaze to gaze(diff index): ", ig) # should be < 0

model = AgentModel(9)
model.load_weights('asterix_test/run8/model') # add path

original_image = data.get_image(I)
image = preprocess_image(tf.convert_to_tensor(original_image),84,84)
saliency = pert.calc_sarfa_saliency_for_image(image,model,mode=MODE)
saliency = pert.image_to_size(saliency)
information_gain = compare.saliency_information_gain(gaze_map, tf.squeeze(saliency,axis=-1))

print("IG gaze to dqn: ", information_gain) # if no weights loaded it is random, e.g. -

fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8,8),label="gaze vs. saliency")
axs[0,0].set_title('Gaze on Salience')
axs[0,0].scatter(gaze_list[:,0],gaze_list[:,1],c='red')
axs[0,0].imshow(saliency)
axs[0,0].axis('off')
axs[0,1].set_title('Image')
axs[0,1].imshow(original_image,cmap='gray')
axs[0,1].axis('off') 
plt.show()