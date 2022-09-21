from my_reader_class import Reader
import perturbation_for_sarfa as pert
import heatmap_comparison as compare

import numpy as np
import scipy.stats as stats 

# choose from which episode to use the data
episodes = ["160_RZ_9166697_Feb-20-16-46-45","167_JAW_2356024_Mar-29-15-42-54","163_RZ_9932825_Mar-01-13-35-12","315_RZ_216627_Jun-10-20-31-25","171_JAW_3395791_Apr-10-16-30-45","243_RZ_593078_Feb-19-10-19-29"]
EPISODE = 5
# choose which samples to use from the episode (make sure you have saved the saliency maps for all of them)
START = 0
LAST = 4000
STEP = 4

FRAME_SKIPS = 4 # how many frames to take at once

print()
print("Test Pixel Binary")
print("------------------")
print()

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[EPISODE] +".txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[EPISODE] +"/")

saliencies = pert.load_saliency(START,LAST,"D:/Documents/Gaze_Data_Project/saliency_database/" + episodes[EPISODE] +"/best/",STEP)
sal_percent_list = [ np.mean(compare.to_binary_flat(sal)) for sal in saliencies ]
sal_percent_1 = np.mean(sal_percent_list)
print("Saliency average pixels that are one: ", sal_percent_1)

gaze_maps = [data.create_gaze_map(i,FRAME_SKIPS) for i in range(START,LAST+1,STEP)]
gaze_percent_list = [ np.mean(compare.to_binary_flat(g)) for g in gaze_maps ]
gaze_percent_1 = np.mean(gaze_percent_list)
print("Gaze maps average pixels that are one: ", gaze_percent_1)

heatmaps = [data.create_gaze_heatmap(i,FRAME_SKIPS) for i in range(START,LAST+1,STEP)]
heatmap_percentage_list = [ np.mean(compare.to_binary_flat(h)) for h in heatmaps ]
heatmap_percent_1 = np.mean(heatmap_percentage_list)
print("Gaze Heatmaps average pixels that are one: ", heatmap_percent_1)

print()
print("Test Pixel Continuous Average Value")
print("-----------------------------------")
print()

sal_percent_list = [ np.mean(sal) for sal in saliencies ]
sal_percent_1 = np.mean(sal_percent_list)
print("Saliency average pixel value: ", sal_percent_1)

only_zeros = np.mean([np.delete(saliency,np.where(saliency !=0)).flatten().shape[0] for saliency in saliencies])
print("Saliency average amount of zero pixels: ", only_zeros)

heatmap_percentage_list = [ np.mean(h) for h in heatmaps ]
heatmap_percent_1 = np.mean(heatmap_percentage_list)
print("Gaze Heatmaps average pixel value: ", heatmap_percent_1)
only_zeros_h = np.mean([np.delete(h,np.where(h != 0)).flatten().shape[0] for h in heatmaps ])
print("Gaze Heatmaps average amount of zero pixels: ", only_zeros_h)