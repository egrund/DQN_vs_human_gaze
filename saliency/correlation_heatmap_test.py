from my_reader_class import Reader
import heatmap_comparison as compare 
import perturbation_for_sarfa as pert

import random as rand
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# choose from which episode to use the data
episodes = ["160_RZ_9166697_Feb-20-16-46-45","167_JAW_2356024_Mar-29-15-42-54"]
EPISODE = 0
# choose which sample to use
I = 100
# you need to at least compare 2 sample to not end in and endless loop in the last part !

FRAME_SKIPS = 4 # How many frames of gaze data to compare to the saliency map

print()
print("Test CC heatmap")
print("------------------")
print()

rand.seed(42)

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[EPISODE] +".txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[EPISODE] +"_extracted/")

saliency = pert.load_saliency(I,path = "D:/Documents/Gaze_Data_Project/saliency_database/" + episodes[EPISODE] +"/run8/")
heatmap = data.create_gaze_heatmap(I,FRAME_SKIPS)

cc = compare.calc_correlation(heatmap, saliency)
print("Correlation: ", cc)
cc_heatmap = compare.heatmap_correlation(heatmap,saliency)

fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(8, 8))

axs[0,0].set_title('SARFA')
axs[0,0].imshow(saliency, cmap="jet")
axs[0,0].axis('off')  

axs[0,1].set_title('Gaze Heatmap')
axs[0,1].imshow(heatmap, cmap="jet")
axs[0,1].axis('off')

axs[0,2].set_title("Correlation")
axs[0,2].imshow(cc_heatmap,cmap="jet")
axs[0,2].axis('off')

plt.tight_layout()
plt.show()