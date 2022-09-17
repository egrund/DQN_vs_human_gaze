from my_reader_class import Reader
import heatmap_comparison as compare 
import perturbation_for_sarfa as pert

import numpy as np
import random as rand
import time

# choose from which episode to use the data
episodes = ["160_RZ_9166697_Feb-20-16-46-45","167_JAW_2356024_Mar-29-15-42-54"]
EPISODE = 0
# choose which samples to use from the episode (make sure you have saved the saliency maps for all of them)
START = 0
LAST = 10 # have 2000
STEP = 1
# you need to at least compare 2 sample to not end in and endless loop in the last part !

FRAME_SKIPS = 4 # How many frames of gaze data to compare to the saliency map

print()
print("Test Mean")
print("------------------")
print()

rand.seed(42)

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[EPISODE] +".txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[EPISODE] +"_extracted/")

start = time.time()

saliencies = pert.load_saliency(START,LAST,"D:/Documents/Gaze_Data_Project/saliency_database/" + episodes[EPISODE] +"/run8/",STEP)
gaze_lists = [data.get_gaze(i,FRAME_SKIPS) for i in range(START,LAST+1,STEP)]
data_maps = [ np.array(gaze_list) for gaze_list in gaze_lists]
heatmaps = [data.create_gaze_heatmap(i,FRAME_SKIPS) for i in range(START,LAST+1,STEP)]

av_distance = []
av_middle = []
for i in range(START,LAST+1,STEP):
    ix = int(i/STEP)
    if not gaze_lists[ix]: # if the human did not look anywhere
        continue
    gaze_list = data_maps[ix]
    saliency = saliencies[int(i/STEP)]
    dif, dif_norm = compare.compare_by_mean(gaze_list, saliency)
    av_distance.append(dif)
    av_middle.append(dif_norm)
    #print(i,"Distance gaze to dqn: ", dif, "Distance to middle: ", dif_norm)

print("Mean distance: ", np.mean(av_distance)," Mean distance to middle: ", np.mean(av_middle))
print("Variance distance: ", np.var(av_distance,ddof=1)," Variance distance to middle: ", np.var(av_middle,ddof=1))

# random
print("Randomly assigned saliency")
av_distance = []
av_middle = []
for i in range(START,LAST+1,STEP):
    ix = int(i/STEP)
    if not gaze_lists[ix]: # if the human did not look anywhere
        continue
    gaze_list = data_maps[ix]
    image_index = rand.randrange(START,LAST+1,STEP)
    saliency = saliencies[int(image_index/STEP)]
    dif, dif_norm = compare.compare_by_mean(gaze_list, saliency)
    av_distance.append(dif)
    av_middle.append(dif_norm)
    #print(i,"RANDOM Distance gaze to dqn: ", dif, "Distance to middle: ", dif_norm)

print("RANDOM Mean distance: ", np.mean(av_distance)," Mean distance to middle: ", np.mean(av_middle))
print("RANDOM Variance distance: ", np.var(av_distance,ddof=1)," Variance distance to middle: ", np.var(av_middle,ddof=1))

print("Randomly assigned gaze heatmap")
av_distance = []
av_middle = []
for i in range(START,LAST+1,STEP):
    ix = int(i/STEP)
    if not gaze_lists[ix]: # if the human did not look anywhere
        continue
    gaze_list = data_maps[ix]
    image_index = rand.randrange(START,LAST+1,STEP)
    while not gaze_lists[ix] or i == image_index:
        image_index = rand.randrange(START,LAST+1,STEP)

    saliency = heatmaps[ix]
    dif, dif_norm = compare.compare_by_mean(gaze_list, saliency)
    av_distance.append(dif)
    av_middle.append(dif_norm)
    #print(i,"RANDOM Distance gaze to dqn: ", dif, "Distance to middle: ", dif_norm)

print("RANDOM Mean distance: ", np.mean(av_distance)," Mean distance to middle: ", np.mean(av_middle))
print("RANDOM Variance distance: ", np.var(av_distance,ddof=1)," Variance distance to middle: ", np.var(av_middle,ddof=1))

end = time.time()
print("Time needed: ",end - start)