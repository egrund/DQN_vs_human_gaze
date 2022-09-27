# Using human gaze data for training a DQN to play Atari games
DRL Project SS22

This is our final project for the course 'Deep Reinforcement Learning' given in SS22 at Osnabrück University.

We used the Atari-HEAD dataset [1] to compare human attention maps and DQN saliency maps, and explore the possibility of adding human attention data to the training process. We have also experimented with making inputs to the DQN more similar to human perceptual input. Further information can be found in the project report.

Our project is divided into four parts:

1. Training a baseline DQN to play Asterix (Atari):
    * Training scripts can be found in the folder [asterix](asterix). 
    * The [weights](dqn_weights) of the best-performing model are also included.

2. Creating saliency maps for the baseline DQN 
    * The scripts for creating saliency maps and analysing them are in [saliency](saliency).

3. Giving gaze data to the DQN
    * Model files are in the folder [gaze_network](gaze_network). 

4. Making the learning more human like by foveating the input and focusing agent's attention on one region at a time. 
    * Code can be found in the folder [asterix_with_blurr](asterix_with_blurr):

In the root folder, there are the files for reading in and processing the gaze data. 
* [my_reader_class.py](my_reader_class.py) is used to read in the data and save it in an instance of `Reader` to be able to access it easily. The object then also has methods to create fixation maps and gaze heatmaps to compare them with saliency maps. 
* For reading in gaze coordinates from the Atari-HEAD files, the file [data_reader.py](data_reader.py) is used which was shared by the creators of the Atari-HEAD dataset and can be found in their [GitHub repository](https://github.com/corgiTrax/Gaze-Data-Processor). 
* The file [create_heatmaps.py](create_heatmaps.py) is used to create heatmaps for training the gaze prediction network, which is then used to augment the DQN for the third part of our project.


[1] Zhang, R., Walshe, C., Liu, Z., Guan, L., Muller, K. S., Whritner, J. A., Zhang, L., Hayhoe, M., & Ballard, D. (2019). *Atari-HEAD: Atari Human Eye-Tracking and Demonstration Dataset (Version 4).* Zenodo. <https://doi.org/10.5281/zenodo.3451402>
