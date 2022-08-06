from my_reader_class import Reader
import heatmap_comparison as compare 
import perturbation_for_sarfa as pert
from sample_trajectory import preprocess_image
from dqn import DQN
from sarfa_saliency import computeSaliencyUsingSarfa


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

data = Reader()

# compare human gaze heatmap with sarfa saliency heatmap
# check perturbation_sarfa_example.py
I = 500
MODE = 'blurred'
gaze_map = data.create_gaze_map(I)
heatmap = data.create_gaze_heatmap(I)

ig = compare.saliency_information_gain(gaze_map,heatmap)
print("IG gaze to gaze(same index): ", ig)

model = DQN(9)
model(tf.random.uniform(shape=(1,84,84,4)),training = False)
#model.load_weights() # add path

image = preprocess_image(tf.convert_to_tensor(data.get_image(I)),84,84)
saliency, perturbed_image = pert.calc_saliency_for_image(image,model,mode=MODE)
saliency = pert.image_to_size(saliency)
information_gain = compare.saliency_information_gain(gaze_map, saliency)

print("IG gaze to dqn: ", information_gain)