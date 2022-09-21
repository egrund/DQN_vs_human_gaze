from my_reader_class import Reader
import heatmap_comparison as compare 
import perturbation_for_sarfa as pert

import random as rand
import time
import numpy as np
import scipy.stats as stats 
from imageio.v2 import imwrite
import pandas as pd

start = time.time()

# choose with what method to compare:
modes = ["TP","TPTN","AUC","IG","CC"] 
MODE = # changed file to do all in for loop otherwise change for loop
# choose what data from the human to use
# AUC + fixation = AUC-Judd
# IG -> fixation
# CC -> heatmap
# rest choose
data_modes = ["fixation","heatmap"]
MODE_DATA = 0 # all in for loop now
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

df = pd.DataFrame()

for m,mode in enumerate(modes):
    for md,mode_data in enumerate(data_modes):
        # remove not working matches: 
        if((m==3 and md == 1) or (m==4 and md == 0)):
            continue

        print()
        print("Test ",mode," using ", mode_data)
        print("------------------")
        print()

        rand.seed(42)
        np.random.seed(42)
        functions = [compare.heatmap_comparison_percentage_saliency_also_true,compare.heatmap_comparison_percentage_same,compare.heatmap_comparison_using_AUC,compare.saliency_information_gain,compare.calc_correlation]

        values = []
        valuesr1 = []
        valuesr2 = []
        for e in range(EPISODE,E_END):
            print(episodes[e])
            data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[e ] +".txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[e ] +"/")
            data_loaders = [data.create_gaze_map, data.create_gaze_heatmap]

            saliencies = pert.load_saliency(START,LAST,"D:/Documents/Gaze_Data_Project/saliency_database/" + episodes[e ] +"/best/",STEP)
            gaze_lists = [data.get_gaze(i,FRAME_SKIPS) for i in range(START,LAST+1,STEP)]
            data_maps = [ data_loaders[md](i,FRAME_SKIPS) for i in range(START,LAST+1,STEP)]

            # save the shuffled saliency to see them (one time execution is enough, see database)
            #for i in range(START,LAST+1,STEP):
                #saliency = saliencies[int(i/STEP)]
                #sal_flat = saliency.flatten()
                #np.random.shuffle(sal_flat)
                #saliency = sal_flat.reshape(saliency.shape)
                #path = "saliency_database/" + episodes[e] +"/shuffled/" + str(i) + "-" + str(i+FRAME_SKIPS-1) + ".png"
                #saliency_to_save = (saliency * 255)
                #imwrite(path,saliency_to_save.astype(np.uint8))

            saliencies_shuffled = pert.load_saliency(START,LAST,"D:/Documents/Gaze_Data_Project/saliency_database/" + episodes[e ] +"/shuffled/",STEP)
            

            # heatmaps for part 3 with comparing to random heatmap
            heatmaps = data_maps
            if(mode_data != "heatmap"):
                heatmaps = [data.create_gaze_heatmap(i,FRAME_SKIPS) for i in range(START,LAST+1,STEP)]

            list_values = []

            for i in range(START,LAST+1,STEP):
                ix = int(i/STEP)
                if not gaze_lists[ix]: # if the human did not look anywhere
                    continue
                heatmap = data_maps[ix]
                saliency = saliencies[ix]
                value = functions[m](heatmap, saliency)
                list_values.append(value)
                #print(i,modes[MODE],value)

            print("Mean: ", np.mean(list_values))
            print("Variance: ",np.var(list_values,ddof=1))
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
                value = functions[m](heatmap, saliency)
                list_valuesr1.append(value)
                #print(i,modes[MODE]," RANDOM ",value)

            print("RA Mean: ", np.mean(list_valuesr1))
            print("RA Variance: ",np.var(list_valuesr1,ddof=1))
            valuesr1.extend(list_valuesr1)

            print("Same saliency but shuffled")

            list_valuesr2 = []
            for i in range(START,LAST+1,STEP):
                ix = int(i/STEP)
                if not gaze_lists[ix]: # if the human did not look anywhere
                    continue
                heatmap = data_maps[ix]

                #saliency = saliencies[ix]
                #sal_flat = saliency.flatten()
                #np.random.shuffle(sal_flat)
                #saliency = sal_flat.reshape(saliency.shape)
                saliency = saliencies_shuffled[ix]
                value = functions[m](heatmap, saliency)
                list_valuesr2.append(value)
                #print(i,modes[MODE]," RANDOM ",value)

            print("Shuffled Mean: ", np.mean(list_valuesr2))
            print("Shuffled Variance: ",np.var(list_valuesr2,ddof=1))
            valuesr2.extend(list_valuesr2)

        # put in dataframe
        print_mode = mode + mode_data
        df[print_mode] = values
        df[print_mode + " RA"] = valuesr1
        df[print_mode + " Shuffled"] = valuesr2


        print()
        print("All Episodes")
        mean = np.mean(values)
        meanr1 = np.mean(valuesr1)
        meanr2 = np.mean(valuesr2)
        confidence = stats.norm.interval(confidence=0.95, loc=mean,scale=stats.sem(values))
        confidencer1 = stats.norm.interval(confidence=0.95, loc=meanr1,scale=stats.sem(valuesr1))
        confidencer2 = stats.norm.interval(confidence=0.95, loc=meanr2,scale=stats.sem(valuesr2))
        print("Mean: ", mean)
        print("Variance: ",np.var(values,ddof=1))
        print("Confidence interval: ",confidence)
        print("+/-", mean - confidence[0])
        print("Randomly assigned saliency")
        print("RA Mean: ", meanr1)
        print("RA Variance: ",np.var(valuesr1,ddof=1))
        print("Confidence interval: ", confidencer1)
        print("+/-", meanr1 - confidencer1[0])
        print("Same saliency but shuffled")
        print("Shuffled Mean: ", meanr2)
        print("Shuffled Variance: ",np.var(valuesr2,ddof=1))
        print("Confidence interval: ",confidencer2)
        print("+/-", meanr2 - confidencer2[0])


        # statistical p calculation
        print("z-test Analysis")
        print("dif to Ra: ", compare.z_test(values, valuesr1) )
        print("dif Shuffled: ", compare.z_test(values, valuesr2) )

        print("greater Ra: ", compare.z_test(values,valuesr1,"greater") )
        print("greater Shuffled: ", compare.z_test(values,valuesr2,"greater") )

        print("smaller Ra: ", compare.z_test(values,valuesr1,"less") )
        print("smaller Shuffled: ", compare.z_test(values,valuesr2,"less") )


df.to_csv("saliency_results/list_all_result_values.csv")
print()
end = time.time()
print("Time needed: ",end - start)