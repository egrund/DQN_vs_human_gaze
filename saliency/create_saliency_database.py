import perturbation_for_sarfa as pert
from my_reader_class import Reader
from sample_trajectory import preprocess_image
from model import AgentModel

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from imageio.v2 import imwrite
from scipy import ndimage as ndi 

start = time.time()

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/") 
#data = Reader(file_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/")

FRAME_SKIPS = 4
I_MAX = data.get_number_frames()
MODE = 'image' #'blurred' # 'black', 'white', 'random'
SIGMA = 5 # size of perturbation
FRAMES_START = 0
FRAMES_END = 1000
SIGMA_SALIENCY = 0.75 # because calculate saliency only for every second pixel

model = AgentModel(9)
model.load_weights('asterix_test/run8/model') # add path

image = preprocess_image(tf.convert_to_tensor(data.get_image(0)),84,84)
masks = pert.create_masks(image,sigma=SIGMA,step=2)
perturbation = tf.zeros(shape=image.shape) # save 10 seconds per image by only creating this one

for i in range(FRAMES_START,FRAMES_END):
	print(i)
	
	images = [preprocess_image(tf.convert_to_tensor(data.get_image(j)),84,84) for j in range(i,i+4)]

	# 1 map
	saliency = pert.calc_sarfa_saliency_for_image(images,model,mode=MODE,masks=masks, perturbation = perturbation)
	saliency = ndi.gaussian_filter(ndi.gaussian_filter(saliency, sigma=SIGMA_SALIENCY), sigma=SIGMA_SALIENCY)
	saliency = pert.image_to_size(saliency/tf.reduce_max(saliency))

	# 4 maps
	#saliencies = pert.calc_sarfa_saliency_for_each_image(images,model,mode=MODE,masks=masks, perturbation = perturbation)
	#saliencies = [ndi.gaussian_filter(ndi.gaussian_filter(saliency, sigma=SIGMA_SALIENCY),sigma=SIGMA_SALIENCY) for saliency in saliencies]
	#saliencies = [pert.image_to_size(saliency) for saliency in saliencies]
	
	# save image
	path = "saliency_database_dif/run8/" + str(i) + "-" + str(i+3) + ".png"
	imwrite(path,saliency)
	#for j in range(FRAME_SKIPS):
	#	imwrite(path + "_" + str(j) + ".png", saliencies[j])

	# save image on baseline image
	original_image = tf.convert_to_tensor(data.get_image(i+4,mode="F"))
	image_saliency = (tf.squeeze(saliency,axis=-1) + original_image/255).numpy()

	path = "saliency_database_dif/run8_on_image/colourful" + str(i) + "-" + str(i+3) + ".png"
	imwrite(path,image_saliency)

end = time.time()
print("Time needed in seconds: ", end - start)