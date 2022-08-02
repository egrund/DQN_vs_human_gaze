from data_reader import read_gaze_data_csv_file as read_gaze
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from scipy import ndimage as ndi 

class Reader :
    """ gets a directory and a file to open a csv as txt file and reads in all the data by using data_reader.py """

    def __init__(self,folder_dir = "C:/Users/eosan/Documents/Gaze_Data/asterix/", file_name = "160_RZ_9166697_Feb-20-16-46-45"):
        self.folder_dir = folder_dir
        self.file_name = file_name
        self.full_name = folder_dir + file_name

        self.frameid2pos, self.frameid2action, self.frameid2duration, self.frameid2unclipped_reward, self.frameid2episode, self.frameid2score, self.frameid_list = read_gaze(self.full_name + ".txt")

        self.images_dir = self.full_name + "_extracted/" + self.file_name[4:15]
        self.x_dim_pic = 160
        self.y_dim_pic = 210

    # getter and setter

    def get_number_frames(self):
        """ returns the number of frames """

        return len(self.frameid_list)

    def get_image(self,i):
        """returns the images of index i in trial"""

        return imread(self.images_dir + str(i) + ".png", pilmode = 'F')

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
        
    def plot_gaze_heatmap(self,i):
        """plots a heatmap of the gaze data of one frame"""

        heatmap = self.create_gaze_heatmap(i)
        fig = plt.figure(figsize = (7, 7))
        plt.imshow(heatmap, cmap = "jet")
        plt.show()



