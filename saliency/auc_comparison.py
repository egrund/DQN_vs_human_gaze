from my_reader_class import Reader
import heatmap_comparison as compare 
import perturbation_for_sarfa as pert
from model import AgentModel

from imageio.v2 import imread
import matplotlib.pyplot as plt
import tensorflow as tf
import random as rand
from scipy import ndimage as ndi 
import time

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/")
#model = AgentModel(9)
#model.load_weights('asterix_test/run8/model')
#model_random = AgentModel(9)

# compare human gaze heatmap with sarfa saliency heatmap

ANZAHL = 1000

start = time.time()

av_auc = 0
for i in range(0,ANZAHL,1):
    if not data.get_gaze(i): # if the human did not look anywhere
        continue
    heatmap = data.create_gaze_heatmap(i) # starts with 0 now
    saliency = imread("D:/Documents/Gaze_Data_Project/saliency_database/run8/" + str(i) + ".png")
    saliency = ndi.gaussian_filter(saliency, sigma=0.75)
    auc, map1, map2 = compare.heatmap_comparison_using_AUC(heatmap, saliency)
    av_auc += auc
    #print(i,"AUC gaze to dqn: ", auc)

auc = av_auc / ANZAHL
print("Average AUC: ", auc)

# compare random matches
av_auc = 0
for i in range(0,ANZAHL,1):
    if not data.get_gaze(i): # if the human did not look anywhere
        continue
    heatmap = data.create_gaze_heatmap(i)
    image_index = rand.randint(0,ANZAHL)
    saliency = imread("D:/Documents/Gaze_Data_Project/saliency_database/run8/" + str(image_index) + ".png")
    saliency = ndi.gaussian_filter(saliency, sigma=0.75)
    auc, map1, map2 = compare.heatmap_comparison_using_AUC(heatmap, saliency)
    av_auc += auc
    #print(i,"RANDOM AUC gaze to dqn: ", auc)

auc = av_auc / ANZAHL
print("Average RANDOM AUC: ", auc)

# compare random model??

end = time.time()
print("Time needed: ",end - start)
