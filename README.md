# DQN_vs_human_gaze
DRL Project SS22 <br />

This is our final project for the course 'Deep Reinforcement Learning' given in SS22 at Osnabrück University. <br />

We used the Atari-HEAD dataset to compare human attention maps and DQN attion maps and tried to improve the learning process by adding human attention data to the algorithm and also by making the learning process more human-like. More information can be found in our report. <br />

Our Project is divided into 4 parts: <br />

1. Training a baseline DQN learning to play Asterix (Atari):
    * The Files for training can be found in the folder [asterix](asterix). 
    * We also uploaded the [weights](best) of our final model.

2. Creating saliency maps for the baseline DQN 
    * The Files for the training are in [saliency](saliency).

3. Giving gaze data to the DQN
    * The Files can be found in the folder [gaze_network](gaze_network). 

4. Making the learning more human like by foveating the input and restricting to one attention area. 

In this main folder, there are the files we need for reading in the data and processing it. 
* [my_reader_class.py](my_reader_class.py) is used to read in the data and save it in an instance of Reader to be able to access it easily. The object then also has methods to create fixation maps and gaze heatmaps to compare them with saliency maps. 
* For actually reading in the data from the files it uses the file [data_reader.py](data_reader.py) which was given from the creators of the Atari-HEAD dataset and can be found on their [Github](https://github.com/corgiTrax/Gaze-Data-Processor). 
* The file [create_heatmaps.py](create_heatmaps.py) is used to create heatmaps for training the gaze prediction network, which is then used to augment the DQN for the third part of our project. 
