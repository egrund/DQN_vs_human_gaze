# creates a database of images with saliency on the original images

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

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/") 

I_MAX = data.get_number_frames()
FRAMES_START = 0
FRAMES_END = 1000
start = time.time()

for i in range(FRAMES_START,FRAMES_END,1):
	print(i)
	original_image = tf.convert_to_tensor(data.get_image(i,mode="F"))
	saliency = imread("D:/Documents/Gaze_Data_Project/saliency_database/run8/" + str(i) + ".png")
	saliency = ndi.gaussian_filter(saliency, sigma=0.75)
	
	image_saliency = (saliency + original_image).numpy()
	max_val = np.max(image_saliency)
	
	# save image
	path = "saliency_database/run8_on_image/" + str(i) + ".png"
	imwrite(path,image_saliency)
	
	# check if worked
	#saliency_new = imread(path)
	#plt.imshow(saliency_new)
	#plt.show()

end = time.time()
print("Time needed: ",end - start)
