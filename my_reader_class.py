from data_reader import read_gaze_data_csv_file as read_gaze
import numpy as np
import matplotlib.pyplot as plt

class Reader :
    """ gets a directory and a file to open a csv as txt file and reads in all the data by using data_reader.py """

    def __init__(self,folder_dir = "C:/Users/eosan/Documents/Gaze_Data/asterix/", file_name = "160_RZ_9166697_Feb-20-16-46-45.txt"):
        self.folder_dir = folder_dir
        self.file_name = file_name
        full_name = folder_dir + file_name

        self.frameid2pos, self.frameid2action, self.frameid2duration, self.frameid2unclipped_reward, self.frameid2episode, self.frameid2score, self.frameid_list = read_gaze(full_name)


    # getter and setter
    def get_number_frames(self):
        """ returns the number of frames """

        return len(self.frameid_list)

    def plot_gaze(self,i):
        """scatters the gaze data of one frame i"""

        as_array = np.array(self.frameid2pos[self.frameid_list[i]])
        # 160x210 picture size
        plt.scatter(as_array[:,0], as_array[:,1])
        plt.show()

    def create_gaze_as_heatmap(self,i):
        """ creates a heatmap out of the gaze data of frame i """



