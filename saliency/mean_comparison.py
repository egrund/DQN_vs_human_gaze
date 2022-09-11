from my_reader_class import Reader
import heatmap_comparison as compare 
import perturbation_for_sarfa as pert
from sample_trajectory import preprocess_image
from model import AgentModel

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
import random as rand
from scipy import ndimage as ndi 

# two random images
# data = Reader(file_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/")
data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/")
# compare human gaze points with sarfa saliency heatmap

ANZAHL = 1000

av_distance = 0
av_middle = 0
for i in range(0,ANZAHL,1):
    if not data.get_gaze(i): # if the human did not look anywhere
        continue
    gaze_list = np.array(data.get_gaze(i)) # starts with 0 now
    saliency = imread("D:/Documents/Gaze_Data_Project/saliency_database/run8/" + str(i) + ".png")
    saliency = ndi.gaussian_filter(saliency, sigma=0.75)

    dif, dif_norm = compare.compare_by_mean(gaze_list, saliency)# tf.squeeze(saliency,axis=-1))
    av_distance += dif
    av_middle += dif_norm
    #print(i,"Distance gaze to dqn: ", dif, "Distance to middle: ", dif_norm)

av_distance = av_distance / ANZAHL
av_middle = av_middle / ANZAHL
print("Average distance: ", av_distance, " Average distance to middle: ",av_middle)

# random
av_distance = 0
av_middle = 0
for i in range(0,ANZAHL,1):
    if not data.get_gaze(i): # if the human did not look anywhere
        continue
    gaze_list = np.array(data.get_gaze(i)) # starts with 0 now
    image_index = rand.randint(0,ANZAHL)
    saliency = imread("D:/Documents/Gaze_Data_Project/saliency_database/run8/" + str(image_index) + ".png")
    saliency = ndi.gaussian_filter(saliency, sigma=0.75)

    dif, dif_norm = compare.compare_by_mean(gaze_list, saliency)# tf.squeeze(saliency,axis=-1))
    av_distance += dif
    av_middle += dif_norm
    #print(i," RANDOM Distance gaze to dqn: ", dif, "Distance to middle: ", dif_norm)

av_distance = av_distance / ANZAHL
av_middle = av_middle / ANZAHL
print("Average RANDOM distance: ", av_distance, " Average distance to middle: ",av_middle)