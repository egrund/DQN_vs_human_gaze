from my_reader_class import Reader
import heatmap_comparison as compare 
import perturbation_for_sarfa as pert

import random as rand
import time
import numpy as np

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
EPISODE = 1
# choose which samples to use from the episode (make sure you have saved the saliency maps for all of them)
START = 0
LAST = 2000 # have 2000
STEP = 1


print()
print("Test ",modes[MODE])
print("------------------")
print()

rand.seed(42)

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[EPISODE] +".txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[EPISODE] +"_extracted/")

functions = [compare.heatmap_comparison_percentage_saliency_also_true,compare.heatmap_comparison_percentage_same,compare.heatmap_comparison_using_AUC,compare.saliency_information_gain,compare.heatmap_correlation]
data_loaders = [data.create_gaze_map, data.create_gaze_heatmap]

start = time.time()

saliencies = pert.load_saliency(START,LAST,"D:/Documents/Gaze_Data_Project/saliency_database/" + episodes[EPISODE] +"/run8/",STEP)

list_values = []
for i in range(START,LAST+1,STEP):
    if not data.get_gaze(i): # if the human did not look anywhere
        continue
    heatmap = data_loaders[MODE_DATA](i)
    saliency = saliencies[int(i/STEP)]
    value = functions[MODE](heatmap, saliency)
    list_values.append(value)
    #print(i,modes[MODE],value)

print("Mean: ", np.mean(list_values))
print("Variance: ",np.var(list_values,ddof=1))

# compare random matches
print("Randomly assigned saliency")

list_values = []
for i in range(START,LAST+1,STEP):
    if not data.get_gaze(i): # if the human did not look anywhere
        continue
    heatmap = data_loaders[MODE_DATA](i)
    image_index = rand.randrange(START,LAST+1,STEP)
    saliency = saliencies[int(image_index/STEP)]
    value = functions[MODE](heatmap, saliency)
    list_values.append(value)
    #print(i,modes[MODE]," RANDOM ",value)

print("RANDOM Mean: ", np.mean(list_values))
print("RANDOM Variance: ",np.var(list_values,ddof=1))

print("Randomly assigned gaze heatmap")

list_values = []
for i in range(START,LAST+1,STEP):
    if not data.get_gaze(i): # if the human did not look anywhere
        continue
    heatmap = data_loaders[MODE_DATA](i)
    image_index = rand.randrange(START,LAST+1,STEP)
    while not data.get_gaze(image_index) or i == image_index:
        image_index = rand.randrange(START,LAST+1,STEP)
    saliency = data.create_gaze_heatmap(image_index)
    value = functions[MODE](heatmap, saliency)
    list_values.append(value)
    #print(i,modes[MODE]," RANDOM ",value)

print("RANDOM Mean: ", np.mean(list_values))
print("RANDOM Variance: ",np.var(list_values,ddof=1))

end = time.time()
print("Time needed: ",end - start)