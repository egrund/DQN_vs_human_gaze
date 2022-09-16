import perturbation_for_sarfa as pert
from my_reader_class import Reader
from sample_trajectory import preprocess_image
from model import AgentModel

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from imageio.v2 import imwrite

start = time.time()

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project//asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/") 
#data = Reader(file_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/")

FRAME_SKIPS = 4
I_MAX = data.get_number_frames()
MODE = 'image' #'blurred' # 'black', 'white', 'random'
SIGMA = 2.8 # size of perturbation
FRAMES_START = 0
FRAMES_END = 2000

model = AgentModel(9)
model.load_weights('asterix_test/run8/model') # add path

image = preprocess_image(tf.convert_to_tensor(data.get_image(0)),84,84)
masks = pert.create_masks(image,sigma=SIGMA,step_hor=2,step_ver=2)
perturbation = tf.zeros(shape=image.shape) # save 10 seconds per image by only creating this one

for i in range(FRAMES_START,FRAMES_END):
	print(i)
	
	images = [preprocess_image(tf.convert_to_tensor(data.get_image(j)),84,84) for j in range(i,i+4)]

	saliency = pert.calc_sarfa_saliency(images,model,mode=MODE,masks=masks, perturbation = perturbation)
	saliency = pert.image_to_size(saliency/tf.reduce_max(saliency))
	
	# save image
	path = "saliency_database/run8/" + str(i) + "-" + str(i+3) + ".png"
	imwrite(path,saliency)

	# save image on baseline image
	original_image = tf.convert_to_tensor(data.get_image(i+3,mode="F"))
	image_saliency = (tf.squeeze(saliency,axis=-1) + original_image/255).numpy()

	path = "saliency_database/run8_on_image/" + str(i) + "-" + str(i+3) + "on_image.png"
	imwrite(path,image_saliency)

end = time.time()
print("Time needed in seconds: ", end - start)