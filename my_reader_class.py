from data_reader import read_gaze_data_csv_file as read_gaze
import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
from scipy import ndimage as ndi 

class Reader :
    """ gets a directory and a file to open a csv as txt file and reads in all the data by using data_reader.py 
    
    Attributes: 
        file_dir (String: "path/file" ): the csc (txt) file to open
        images_dir (String: "path"): the folder where the images for the trial are
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

    def __init__(self,file_dir, images_dir, image_type = ".png"):
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

    def get_image(self,i,mode='RGB'):
        """returns the images of index i in trial"""

        return imread(self.images_dir + self.frameid_list[i] + self.image_type, pilmode = mode)

    def get_gaze(self,i):
        """returns a list of all gaze positions in frame i """

        gaze_list = self.frameid2pos[self.frameid_list[i]] # attentions, changed indexing. now 0 to end not 1 to end
        if(gaze_list == None):
            return []
        return gaze_list

    def get_action(self,i):
        """returns action of frame i """
        return self.frameid2action[self.frameid_list[i-1]]

    # methods

    def plot_image(self,i):
        """plots images of index i in trial"""

        image = self.get_image(i,'F')
        fig = plt.figure(figsize = (7,7))
        plt.imshow(image,cmap = 'gray')
        plt.show()

    def plot_gaze(self,i):
        """scatters the gaze data of one frame i"""

        gaze = self.get_gaze(i)
        if(gaze == None):
            return
        gaze_list = np.array(gaze)
        
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(gaze_list[:,0], gaze_list[:,1])
        ax.set_xlim([0,self.x_dim_pic])
        # (0,0) is supposed to be upper left
        ax.set_ylim([self.y_dim_pic,0])
        plt.show()

    def create_gaze_map(self,i):
        """ creates a map with 0 everywhere and 1 where the gaze is (rounded) """

        gaze_list = self.get_gaze(i)
        if(gaze_list == None):
            return None
        image = self.get_image(i,'F')
        gaze_map = np.zeros_like(image).T # transposed because we want dim 0 = x and dim 1 = y
        if gaze_list is not None and len(gaze_list) > 0:
            for (x,y) in gaze_list:
                gaze_map[min(self.x_dim_pic,int(x)) -1, min(self.y_dim_pic,int(y))-1] = 1
        return gaze_map.T

    def create_all_gaze_maps(self):
        """ creates all the gaze maps for the trial (for every frame) """
        gaze_maps = list() 
        for i in range(self.get_number_frames()):
            map = self.create_gaze_map(i)
            gaze_maps.append(map)
        return gaze_maps

    def create_gaze_heatmap(self,i):
        """ creates a heatmap out of the gaze data of frame i """
        
        heatmap = self.create_gaze_map(i)
        heatmap = ndi.gaussian_filter(heatmap, sigma=10) # sigma should be one visual degree
        return heatmap 

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
        if(heatmap == None):
            return
        fig = plt.figure(figsize = (7, 7))
        plt.imshow(heatmap, cmap = "jet")
        plt.show()