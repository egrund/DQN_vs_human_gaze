from my_reader_class import Reader
import perturbation_for_sarfa as pert
from model import AgentModel
from heatmap_comparison import round_with_threshold

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imageio.v2 import imread

episodes = ["160_RZ_9166697_Feb-20-16-46-45","167_JAW_2356024_Mar-29-15-42-54"]
EPISODE = 0

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45/")
#data = Reader(file_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/")

I = 40
original_image = tf.convert_to_tensor(data.get_image(I))
model = AgentModel(9)
model.load_weights('asterix_test/best/model')

saliency = pert.load_saliency(I,path="D:/Documents/Gaze_Data_Project/saliency_database/" + episodes[EPISODE] +"/best/")

map_max = tf.reduce_max(saliency)
map_normal = saliency / map_max
binary_map = np.where(map_normal > 0.2, 1, 0)

fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(8, 8))

axs[0,0].set_title('Heatmap')
axs[0,0].imshow(saliency,cmap="jet")
axs[0,0].axis('off')

axs[0,1].set_title('Heatmap scaled')
axs[0,1].imshow(map_normal,cmap="jet")
axs[0,1].axis('off') 

axs[0,2].set_title('binary Heatmap')
axs[0,2].imshow(binary_map,cmap="jet")
axs[0,2].axis('off')  

plt.tight_layout()
plt.show()