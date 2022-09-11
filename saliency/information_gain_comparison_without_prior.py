from my_reader_class import Reader
import heatmap_comparison as compare 
import perturbation_for_sarfa as pert
from sample_trajectory import preprocess_image
from model import AgentModel

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
from scipy import ndimage as ndi 
import random as rand

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/")
#model = AgentModel(9)
#model.load_weights('asterix_test/run8/model') # add path

# compare human gaze points with sarfa saliency heatmap

# similar # should be > 0
# not so similar # should be < 0
ANZAHL = 1000
av_info_gain_sum = 0
for i in range(0,ANZAHL,1):
    gaze_map = data.create_gaze_map(i)

    saliency = imread("D:/Documents/Gaze_Data_Project/saliency_database/run8/" + str(i) + ".png")
    saliency = ndi.gaussian_filter(saliency, sigma=0.75)
    information_gain = compare.saliency_information_gain_without_prior(gaze_map, saliency) # maybe tf squeeze
    av_info_gain_sum += information_gain
    print(i," IG gaze to dqn: ", information_gain) # if no weights loaded it is random, e.g. -

average_gain = av_info_gain_sum / ANZAHL
print("Average Informaiton gain without prior: ", average_gain)


# random information gain
av_info_gain_sum = 0
for i in range(0,ANZAHL,1):
    gaze_map = data.create_gaze_map(i)
    image_index = rand.randint(0,ANZAHL)
    saliency = imread("D:/Documents/Gaze_Data_Project/saliency_database/run8/" + str(image_index) + ".png")
    saliency = ndi.gaussian_filter(saliency, sigma=0.75)
    information_gain = compare.saliency_information_gain_without_prior(gaze_map, saliency) # maybe tf squeeze
    av_info_gain_sum += information_gain
    #print(i," IG RANDOM gaze to dqn: ", information_gain) # if no weights loaded it is random, e.g. -

average_gain = av_info_gain_sum / ANZAHL
print("Average RANDOM Informaiton gain without prior: ", average_gain)