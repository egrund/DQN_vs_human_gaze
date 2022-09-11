import perturbation_for_sarfa as pert
from my_reader_class import Reader
from sample_trajectory import preprocess_image
from model import AgentModel

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from imageio.v2 import imwrite
from imageio.v2 import imread
from scipy import ndimage as ndi 

data = Reader(file_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/") 

FRAME_SKIPS = 4
I_MAX = data.get_number_frames()
MODE = 'image' #'blurred' # 'black', 'white', 'random'
SIGMA = 5 # size of perturbation
FRAMES_START = 1000
FRAMES_END = 1001
SIGMA_SALIENCY = 0.75 # because calculate saliency only for every second pixel

model = AgentModel(9)
model.load_weights('asterix_test/run8/model') # add path

image = preprocess_image(tf.convert_to_tensor(data.get_image(0)),84,84)
masks = pert.create_masks(image,sigma=SIGMA,step=2)
perturbation = tf.zeros(shape=image.shape) # save 10 seconds per image by only creating this one

for i in range(FRAMES_START,FRAMES_END,1):
	print(i)
	#start = time.time()
	
	image = preprocess_image(tf.convert_to_tensor(data.get_image(i)),84,84)
	saliency = pert.calc_sarfa_saliency_for_image(image,model,mode=MODE,masks=masks, perturbation = perturbation, frame_skips=FRAME_SKIPS)
	saliency = ndi.gaussian_filter(saliency, sigma=SIGMA_SALIENCY)
	saliency = pert.image_to_size(saliency)
	# alternatively use my_perturbance_map, because it is way quicker (40 sec less per image when I tried)
	
	# save image
	path = "saliency_database/run8/" + str(i) + ".png"
	imwrite(path,saliency)
	
	#end = time.time()
	#print("Time needed: ",end - start)
	
	# check if worked
	#saliency_new = imread(path)
	#plt.imshow(saliency_new)
	#plt.show()
