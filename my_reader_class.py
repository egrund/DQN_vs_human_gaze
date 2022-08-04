from data_reader import read_gaze_data_csv_file as read_gaze
import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
from scipy import ndimage as ndi 

class Reader :
    """ gets a directory and a file to open a csv as txt file and reads in all the data by using data_reader.py 
    
    Attributes: 
        file_dir (String): the csc (txt) file to open
        images_dir (String): the folder where the images for the trial are
        image_type (String): the image format e.g. .png or .jpg
        x_dim_pic (int): amount of pixels in x direction of the image (horizontal)
        y_dim_pic (int): amount of pixels in y direction of the image (vertical)

        frameid_list (list): a list if all the frame ids (keys for the following dictionaries)
        frameid2pos (dict): dictionary with list of all gaze positions for each frame (might be None)
        frameid2action (dict): dictionary with all the actions taken for each frame
        frameid2duration (dict): how long the person looked at the frame for each frame
        frameid2unlclipped_reward (dict):
        frameid2episode (dict):
        frameid2score (dict): what the current score was in the frame
    """

    def __init__(self,file_dir = "C:/Users/eosan/Documents/Gaze_Data/asterix/160_RZ_9166697_Feb-20-16-46-45.txt", images_dir = "C:/Users/eosan/Documents/Gaze_Data/asterix/160_RZ_9166697_Feb-20-16-46-45_extracted/", image_type = ".png"):
        self.file_dir = file_dir
        self.images_dir = images_dir
        self.x_dim_pic = 160
        self.y_dim_pic = 210
        self.image_type = image_type
        self.frameid2pos, self.frameid2action, self.frameid2duration, self.frameid2unclipped_reward, self.frameid2episode, self.frameid2score, self.frameid_list = read_gaze(self.file_dir)

    # getter and setter

    def get_number_frames(self):
        """ returns the number of frames """

        return len(self.frameid_list)

    def get_image(self,i):
        """returns the images of index i in trial"""

        return imread(self.images_dir + self.frameid_list[i] + self.image_type, pilmode = 'F')

    # methods

    def plot_image(self,i):
        """plots images of index i in trial"""

        image = self.get_image(i)
        fig = plt.figure(figsize = (7,7))
        plt.imshow(image,cmap = 'gray')
        plt.show()

    def plot_gaze(self,i):
        """scatters the gaze data of one frame i"""

        gaze_list = np.array(self.frameid2pos[self.frameid_list[i-1]])
        
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(gaze_list[:,0], gaze_list[:,1])
        ax.set_xlim([0,self.x_dim_pic])
        # (0,0) is supposed to be upper left
        ax.set_ylim([self.y_dim_pic,0])
        plt.show()

    def create_gaze_heatmap(self,i):
        """ creates a heatmap out of the gaze data of frame i """
        
        gaze_list = self.frameid2pos[self.frameid_list[i-1]]
        image = self.get_image(i)
        heatmap = np.zeros_like(image).T # transposed because we want dim 0 = x and dim 1 = y
        if gaze_list is not None and len(gaze_list) > 0:
            for (x,y) in gaze_list:
                heatmap[min(self.x_dim_pic,int(x)) -1, min(self.y_dim_pic,int(y))-1] = 1

        heatmap = ndi.gaussian_filter(heatmap, sigma=10) # sigma should be one visual degree
        return heatmap.T # so x is horizontal and y is vertical

    def create_all_gaze_heatmaps(self):
        """ creates all the gaze heatmaps for the trial (for every frame) """

        heatmaps = list()
        for i in range(self.get_number_frames()):
            heatmap = self.create_gaze_heatmap(i)
            heatmaps.append(heatmap)
        return heatmaps
        
    def plot_gaze_heatmap(self,i):
        """plots a heatmap of the gaze data of one frame"""

        heatmap = self.create_gaze_heatmap(i)
        fig = plt.figure(figsize = (7, 7))
        plt.imshow(heatmap, cmap = "jet")
        plt.show()



