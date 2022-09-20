from my_reader_class import Reader
import heatmap_comparison as compare 
import perturbation_for_sarfa as pert

import random as rand
import time
import numpy as np
import scipy.stats as stats 

# choose with what method to compare:
modes = ["TP","same","AUC","IG","CC"] 
MODE = 2
# choose what data from the human to use
# AUC + fixation = AUC-Judd
# IG -> fixation
# CC -> heatmap
# rest choose
data_modes = ["fixation","heatmap"]
MODE_DATA = 1
# choose from which episode to use the data
episodes = ["160_RZ_9166697_Feb-20-16-46-45","167_JAW_2356024_Mar-29-15-42-54","163_RZ_9932825_Mar-01-13-35-12","315_RZ_216627_Jun-10-20-31-25","171_JAW_3395791_Apr-10-16-30-45","243_RZ_593078_Feb-19-10-19-29"]
EPISODE = 0
E_END = 6
# choose which samples to use from the episode (make sure you have saved the saliency maps for all of them)
START = 0
LAST = 4001
STEP = 4
# you need to at least compare 2 sample to not end in and endless loop in the last part !

FRAME_SKIPS = 4 # How many frames of gaze data to compare to the saliency map

print()
print("Test ",modes[MODE])
print("------------------")
print()

start = time.time()

rand.seed(42)
functions = [compare.heatmap_comparison_percentage_saliency_also_true,compare.heatmap_comparison_percentage_same,compare.heatmap_comparison_using_AUC,compare.saliency_information_gain,compare.calc_correlation]

values = []
valuesr1 = []
valuesr2 = []
for e in range(EPISODE,E_END):
    data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[e ] +".txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[e ] +"/")
    data_loaders = [data.create_gaze_map, data.create_gaze_heatmap]

    saliencies = pert.load_saliency(START,LAST,"D:/Documents/Gaze_Data_Project/saliency_database/" + episodes[e ] +"/best/",STEP)
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
            continue
        heatmap = data_maps[ix]
        saliency = saliencies[ix]
        value = functions[MODE](heatmap, saliency)
        list_values.append(value)
        #print(i,modes[MODE],value)

    print(episodes[e]," Mean: ", np.mean(list_values))
    print(episodes[e]," Variance: ",np.var(list_values,ddof=1))
    values.extend(list_values)

    # compare random matches
    print("Randomly assigned saliency")

    list_valuesr1 = []
    for i in range(START,LAST+1,STEP):
        ix = int(i/STEP)
        if not gaze_lists[ix]: # if the human did not look anywhere
            continue
        heatmap = data_maps[ix]
        image_index = rand.randrange(START,LAST+1,STEP)
        saliency = saliencies[int(image_index/STEP)]
        value = functions[MODE](heatmap, saliency)
        list_valuesr1.append(value)
        #print(i,modes[MODE]," RANDOM ",value)

    print(episodes[e]," RANDOM Mean: ", np.mean(list_valuesr1))
    print(episodes[e]," RANDOM Variance: ",np.var(list_valuesr1,ddof=1))
    valuesr1.extend(list_valuesr1)

    print("Randomly assigned gaze heatmap") # difficult with AUC

    list_valuesr2 = []
    for i in range(START,LAST+1,STEP):
        ix = int(i/STEP)
        if not gaze_lists[ix]: # if the human did not look anywhere
            continue
        heatmap = data_maps[ix]

        image_index = rand.randrange(START,LAST+1,STEP)
        while not gaze_lists[ix] or i == image_index:
            image_index = rand.randrange(START,LAST+1,STEP)

        saliency = heatmaps[int(image_index/STEP)]
        value = functions[MODE](heatmap, saliency)
        list_valuesr2.append(value)
        #print(i,modes[MODE]," RANDOM ",value)

    print(episodes[e]," RANDOM Mean: ", np.mean(list_valuesr2))
    print(episodes[e]," RANDOM Variance: ",np.var(list_valuesr2,ddof=1))
    values.r2.extend(list_valuesr2)

mean = np.mean(values)
meanr1 = np.mean(valuesr1)
meanr2 = np.mean(valuesr2)
print("Mean: ", mean)
print("Variance: ",np.var(values,ddof=1))
print("Confidence interval: ",stats.norm.interval(alpha=0.95, loc=mean,scale=stats.sem(values)))
print("Randomly assigned saliency")
print("RANDOM Mean: ", meanr1)
print("RANDOM Variance: ",np.var(valuesr1,ddof=1))
print("Confidence interval: ",stats.norm.interval(alpha=0.95, loc=meanr1,scale=stats.sem(valuesr1)))
print("Randomly assigned gaze heatmap")
print("RANDOM Mean: ", meanr2)
print("RANDOM Variance: ",np.var(valuesr2,ddof=1))
print("Confidence interval: ",stats.norm.interval(alpha=0.95, loc=meanr2,scale=stats.sem(valuesr2)))


# statistical p calculation
print("t-test Analysis")
print("dif r1: ", compare.z_test(a=values, b=valuesr1) )
print("dif r2: ", compare.z_test(a=values, b=valuesr2) )

print("greater r1: ", compare.z_test(values,valuesr1,"greater") )
print("greater r2: ", compare.z_test(values,valuesr2,"greater") )

print("smaller r1: ", compare.z_test(values,valuesr1,"smaller") )
print("smaller r2: ", compare.z_test(values,valuesr2,"smaller") )

end = time.time()
print("Time needed: ",end - start)