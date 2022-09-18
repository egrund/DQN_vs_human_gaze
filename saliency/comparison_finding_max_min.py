from my_reader_class import Reader
import heatmap_comparison as compare 
import perturbation_for_sarfa as pert

import random as rand
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# choose with what method to compare:
modes = ["TP","same","AUC","IG","CC"] 
MODE = 4
# choose what data from the human to use
# AUC + fixation = AUC-Judd
# IG -> fixation
# CC -> heatmap
# rest choose
data_modes = ["fixation","heatmap"]
MODE_DATA = 1
# choose from which episode to use the data
episodes = ["160_RZ_9166697_Feb-20-16-46-45","167_JAW_2356024_Mar-29-15-42-54"]
EPISODE = 0
# choose which sample to use
START = 0
LAST = 2000 # have 2000
STEP = 1
# you need to at least compare 2 sample to not end in and endless loop in the last part !

FRAME_SKIPS = 4 # How many frames of gaze data to compare to the saliency map

print()
print("Test max min",modes[MODE])
print("-------------------------")
print()

rand.seed(42)

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[EPISODE] +".txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[EPISODE] +"_extracted/")

functions = [compare.heatmap_comparison_percentage_saliency_also_true,compare.heatmap_comparison_percentage_same,compare.heatmap_comparison_using_AUC,compare.saliency_information_gain,compare.calc_correlation]
data_loaders = [data.create_gaze_map, data.create_gaze_heatmap]

saliencies = pert.load_saliency(START,LAST,"D:/Documents/Gaze_Data_Project/saliency_database/" + episodes[EPISODE] +"/run8/",STEP)
gaze_lists = [data.get_gaze(i,FRAME_SKIPS) for i in range(START,LAST+1,STEP)]
data_maps = [ data_loaders[MODE_DATA](i,FRAME_SKIPS) for i in range(START,LAST+1,STEP)]

# heatmaps for part 3 with comparing to random heatmap
heatmaps = data_maps
if(data_modes[MODE_DATA] != "heatmap"):
    heatmaps = [data.create_gaze_heatmap(i,FRAME_SKIPS) for i in range(START,LAST+1,STEP)]

list_values = []
for i in range(START,LAST+1,STEP):
    ix = int(i/STEP)
    if not gaze_lists[ix]: # if the human did not look anywhere
        list_values.append(np.mean(list_values)) # so this is not max or min
        continue
    heatmap = data_maps[ix]
    saliency = saliencies[ix]
    value = functions[MODE](heatmap, saliency)
    list_values.append(value)
    #print(i,modes[MODE],value)

argmax = np.argmax(list_values)
argmin = np.argmin(list_values)

print("Mean: ", np.mean(list_values))
print("Variance: ",np.var(list_values,ddof=1))
print("Max: ", np.max(list_values), "Index: ",argmax)
print("Min: ", np.min(list_values), "Index: ",argmin)

if(modes[MODE] == "CC"):
    ccmax = list_values[argmax]
    ccmin = list_values[argmax]

    cc_heatmap_max = compare.heatmap_correlation(heatmaps[argmax],saliencies[argmax])
    cc_heatmap_min = compare.heatmap_correlation(heatmaps[argmin],saliencies[argmin])

    fig, axs = plt.subplots(nrows=2, ncols=3, squeeze=False, figsize=(8, 8))

    axs[0,0].set_title('SARFA')
    axs[0,0].imshow(saliencies[argmax], cmap="jet")
    axs[0,0].axis('off')  

    axs[0,1].set_title('Gaze Heatmap')
    axs[0,1].imshow(heatmaps[argmax], cmap="jet")
    axs[0,1].axis('off')

    axs[0,2].set_title("Max Correlation")
    axs[0,2].imshow(cc_heatmap_max,cmap="jet")
    axs[0,2].axis('off')

    axs[1,0].set_title('SARFA')
    axs[1,0].imshow(saliencies[argmin], cmap="jet")
    axs[1,0].axis('off')  

    axs[1,1].set_title('Gaze Heatmap')
    axs[1,1].imshow(heatmaps[argmin], cmap="jet")
    axs[1,1].axis('off')

    axs[1,2].set_title("Min Correlation")
    axs[1,2].imshow(cc_heatmap_min,cmap="jet")
    axs[1,2].axis('off')

    plt.tight_layout()
    plt.show()