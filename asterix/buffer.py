import joblib
import random as rand
import numpy as np
from pathlib import Path

class Buffer:
    """
    This class implements a replay buffer capable of using prioritized replay and storing elements in memory and on the hard drive
    """

    def __init__(self, max_buffer_size, min_buffer_size, buffer_path, keep_in_mem, use_prioritized_replay) -> None:

        self.keep_in_mem = keep_in_mem
        self.buffer_path = buffer_path

        # create all necessary folders if data points should be stored on hard drive
        if not self.keep_in_mem:
            Path(buffer_path).mkdir(parents=True, exist_ok=True)

            # each directory will only store a maximum of 10000 files, thus we have to create a sub directory for each batch of 10000 data points
            for i in range(int(max_buffer_size/10000)+1):
                Path(buffer_path + str(i)).mkdir(parents=True, exist_ok=True)

        self.at_file = 0

        self._max_buffer_size = max_buffer_size
        self._min_buffer_size = min_buffer_size
        self.use_prioritized_replay = use_prioritized_replay
        self.priorities = {}
        self.data = {}

    def extend(self, data):
        """
        Add new data to the buffer

        Args:
            - data <List<Object>> : List of data points that should be added
        """

        # case: prioritized replay should be used and the buffer has been fully filled yet
        if self.use_prioritized_replay and not len(self.data.keys()) < self._max_buffer_size:

            # get the maximum priority. This will be used as the initial priority for new data points
            keys = list(self.priorities.keys())
            priorities = list(self.priorities.values())
            max_priority = max(priorities)

            # sort the data to replace data points with a low priority
            for sars, index in zip(data, [int(key) for key,_ in sorted(zip(keys,priorities), key = lambda item : item[1])]):

                self.at_file = index
                
                if self.keep_in_mem:
                    self.data[str(self.at_file)] = sars
                else:
                    self.data[str(self.at_file)] = "0"

                    # write data to file
                    with open( self.buffer_path + str(int(self.at_file/10000)) + "/" + str(self.at_file) + ".pkl", "wb" ) as f:
                        joblib.dump(sars,f)

                self.priorities[str(self.at_file)] = max_priority
                
        else:

            for sars in data:

                if self.keep_in_mem:
                    self.data[str(self.at_file)] = sars
                else:
                    self.data[str(self.at_file)] = "0"

                    # write data to file
                    with open( self.buffer_path + str(int(self.at_file/10000)) + "/" + str(self.at_file) + ".pkl", "wb" ) as f:
                        joblib.dump(sars,f)

                # set initial priority as 100
                self.priorities[str(self.at_file)] = 100

                # skip to the next buffer element and begin at 0 if necessary
                self.at_file += 1
                if self.at_file > self._max_buffer_size:
                    self.at_file = 0

    def get_indices(self, batch_size):
        """
        Get indices for sampling elements from the buffer

        Args:
            - batch size <int> : How many indices should be sample

        Returns
            - <List<int>> : indices of the elements which should be sampled
        """

        # normalize and scale priorities
        if self.use_prioritized_replay:
            priorities = np.array(list(self.priorities.values()))
            priorities = np.power(priorities,0.3)
            priorities = priorities/(np.sum(priorities)+1)

            return rand.choices([int(key) for key in self.priorities.keys()], weights=priorities,k = batch_size)
        else: 
            return rand.choices([int(key) for key in self.priorities.keys()],k = batch_size)
    
    def sample_minibatch(self, indices):
        """
        Get the elements for the given indices obtained from get_indices

        Args:
            - indices <List<int>> : indices of the elements which should be sampled
        Returns
            - <dict> : Dictionary containing the elements from the buffer as well as the indices 
        """
        
        s_a_r_s = []

        for index in indices:

            if self.keep_in_mem:
                s_a_r_s.append(self.data[str(index)])
            else:
                # load elements from hard drive
                with open( "./buffer/" + str(int(index/10000)) + "/" + str(index) + ".pkl", "rb" ) as f:
                    s_a_r_s.append(joblib.load(f))

        s = [elem[0] for elem in s_a_r_s]
        a = [elem[1] for elem in s_a_r_s]
        r = [elem[2] for elem in s_a_r_s]
        s_new = [elem[3]  for elem in s_a_r_s]
        done = [elem[4] for elem in s_a_r_s]

        return {"values" : (s,a,r,s_new,done), "indices" : indices}


    def update_priorities(self,priorities):
        """
        Update the priorities for the elements in 'priorities'"

        Args:
            - priorities <Dict> : Dictionary containing the indice as well as the new priority for an element in the buffer
        """

        for key, value in priorities.items():
            self.priorities[key] = value
        