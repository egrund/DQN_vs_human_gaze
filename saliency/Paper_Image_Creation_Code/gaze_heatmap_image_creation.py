from my_reader_class import Reader
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

data = Reader(file_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "D:/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45/")
#data = Reader(file_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "/media/egrund/Storage/Documents/Gaze_Data_Project/asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/")

I = 20
original_image = data.get_image(I)
gaze_list = np.array(data.get_gaze(I))
gaze_heatmap = data.create_gaze_heatmap(I)
fixation_map = data.create_gaze_map(I)

fig, axs = plt.subplots(nrows=1, ncols=4, squeeze=False, figsize=(8, 8))

axs[0,0].set_title('Image')
axs[0,0].imshow(original_image)
axs[0,0].axis('off')

axs[0,1].set_title('Gaze Locations on Image')
axs[0,1].imshow(original_image)
axs[0,1].scatter(gaze_list[:,0],gaze_list[:,1],c="lightgreen")
axs[0,1].axis('off') 

axs[0,2].set_title('Fixation Map')
axs[0,2].imshow(fixation_map,cmap="jet")
axs[0,2].axis('off')  

axs[0,3].set_title('Gaze Heatmap')
axs[0,3].imshow(gaze_heatmap,cmap="jet")
axs[0,3].axis('off') 

plt.tight_layout()
plt.show()
