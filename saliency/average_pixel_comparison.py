from my_reader_class import Reader
import perturbation_for_sarfa as pert
import heatmap_comparison as compare

import numpy as np
import scipy.stats as stats 

# choose from which episode to use the data
episodes = ["160_RZ_9166697_Feb-20-16-46-45","167_JAW_2356024_Mar-29-15-42-54"]
EPISODE = 0
# choose which samples to use from the episode (make sure you have saved the saliency maps for all of them)
START = 0
LAST = 2000 # have 2000
STEP = 1

FRAME_SKIPS = 4 # how many frames to take at once

print()
print("Test Pixel Binary")
print("------------------")
print()

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[EPISODE] +".txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[EPISODE] +"_extracted/")

saliencies = pert.load_saliency(START,LAST,"D:/Documents/Gaze_Data_Project/saliency_database/" + episodes[EPISODE] +"/run8/",STEP)
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

# make p value to check if really so different
print("t-test Analysis")
print("binary saliency dif fixation map: ", compare.t_test(a=sal_percent_list, b=gaze_percent_list) )
print("binary saliency dif  heatmap: ", compare.t_test(a=sal_percent_list, b=heatmap_percentage_list) )
print("binary saliency greater fixation map: ", compare.t_test(sal_percent_list,gaze_percent_list,"greater") )
print("binary saliency greater  heatmap: ", compare.t_test(sal_percent_list, heatmap_percentage_list,"greater")  )


print()
print("Test Pixel Continuous Average Value")
print("-----------------------------------")
print()

sal_percent_list = [ np.mean(sal) for sal in saliencies ]
sal_percent_1 = np.mean(sal_percent_list)
print("Saliency average pixels that are one: ", sal_percent_1)

heatmap_percentage_list = [ np.mean(h) for h in heatmaps ]
heatmap_percent_1 = np.mean(heatmap_percentage_list)
print("Gaze Heatmaps average pixels that are one: ", heatmap_percent_1)


# make p value to check if really so different
print("t-test Analysis")
print("continuous saliency dif heatmap: ", compare.t_test(a=sal_percent_list, b=heatmap_percentage_list) )
print("continuous : ", compare.t_test(sal_percent_list, heatmap_percentage_list,"greater") )