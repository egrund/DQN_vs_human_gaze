from my_reader_class import Reader
import heatmap_comparison as compare 
import perturbation_for_sarfa as pert
from sample_trajectory import preprocess_image
from model import AgentModel

import matplotlib.pyplot as plt
import tensorflow as tf

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/")

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
MODE = 'black'
heatmap = data.create_gaze_heatmap(I)
model = AgentModel(9)
model.load_weights('asterix_test/run8/model') # add path
print("Shape gazemap: ", heatmap.shape)

image = preprocess_image(tf.convert_to_tensor(data.get_image(I)),84,84)
saliency = pert.calc_sarfa_saliency_for_image(image,model,mode=MODE)
saliency = pert.image_to_size(saliency)

print("Saliency Shape: ",saliency.shape)

auc, map1, map2 = compare.heatmap_comparison_using_AUC(heatmap, saliency)

print("AUC gaze to dqn: ", auc)
fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 8),label="maps: gaze to dqn")
axs[0,0].set_title('Map1 (Gaze)')
axs[0,0].imshow(map1)
axs[0,0].axis('off')
axs[0,1].set_title('Map2 (Sarfa)')
axs[0,1].imshow(map2)
axs[0,1].axis('off') 
plt.show()
