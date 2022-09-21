import perturbation_for_sarfa as pert
from my_reader_class import Reader
from model import AgentModel

import numpy as np
import tensorflow as tf
import time
from imageio.v2 import imwrite

episodes = ["160_RZ_9166697_Feb-20-16-46-45","167_JAW_2356024_Mar-29-15-42-54","163_RZ_9932825_Mar-01-13-35-12","315_RZ_216627_Jun-10-20-31-25","171_JAW_3395791_Apr-10-16-30-45","243_RZ_593078_Feb-19-10-19-29"]
E = 5
E_ENDE = 6

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[E] +".txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[E] + "/") 

FRAME_SKIPS = 4
#I_MAX = data.get_number_frames()
MODE = 'image' #'blurred' # 'black', 'white', 'random'
SIGMA = 2.8 # size of perturbation
FRAMES_START = 0
FRAMES_END = 2788
STEP = 4

model = AgentModel(9)
model.load_weights('asterix_test/best/model') # add path

image = pert.preprocess_image(tf.convert_to_tensor(data.get_image(0)))

masks = pert.create_masks(image,sigma=SIGMA,step_hor=2,step_ver=2)
perturbation = tf.zeros(shape=image.shape) # saves a lot per image by only creating this one here once vs for every perturbed image

start = time.time()
for e in range(E,E_ENDE):
	data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[e] +".txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/" + episodes[e] + "/") 
	for i in range(FRAMES_START,FRAMES_END,STEP):
		print(i)
		
		images = [pert.preprocess_image(tf.convert_to_tensor(data.get_image(j))) for j in range(i,i+FRAME_SKIPS)]

		saliency = pert.calc_sarfa_saliency(images,model,mode=MODE,masks=masks, perturbation = perturbation)
		saliency = pert.image_to_size(saliency/tf.reduce_max(saliency))
		
		# save image
		path = "saliency_database/" + episodes[e] +"/best/" + str(i) + "-" + str(i+FRAME_SKIPS-1) + ".png"
		saliency_to_save = (saliency * 255)
		imwrite(path,saliency_to_save.numpy().astype(np.uint8))

		# save image on baseline image
		original_image = tf.convert_to_tensor(data.get_image(i+FRAME_SKIPS-1,mode="F"))
		image_saliency = (tf.squeeze(saliency_to_save,axis=-1) + original_image).numpy().astype(np.uint8)

		path = "saliency_database/" + episodes[e] + "/best_on_image/" + str(i) + "-" + str(i+FRAME_SKIPS-1) + "on_image.png"
		imwrite(path,image_saliency)
	print("done with ", episodes[e])

end = time.time()
print("Time needed in seconds: ", end - start)