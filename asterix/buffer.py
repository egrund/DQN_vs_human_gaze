from multiprocessing.managers import SyncManager
import joblib
import os
import random as rand
import numpy as np
from pathlib import Path
class Buffer:

    def __init__(self, max_buffer_size, min_buffer_size, buffer_path, keep_in_mem, use_prioritized_replay) -> None:

        self.keep_in_mem = keep_in_mem
        self.buffer_path = buffer_path

        Path(buffer_path).mkdir(parents=True, exist_ok=True)

        for i in range(int(max_buffer_size/10000)+1):
            Path(buffer_path + str(i)).mkdir(parents=True, exist_ok=True)

        self.at_file = 0

        self._max_buffer_size = max_buffer_size
        self._min_buffer_size = min_buffer_size
        self.use_prioritized_replay = use_prioritized_replay
        self.last_batch_indexes = list()
        
        self.priorities = {}
        self.data = {}


    def extend(self, data):

        if self.use_prioritized_replay and not len(self.data.keys()) < self._max_buffer_size:
            keys = list(self.priorities.keys())
            priorities = list(self.priorities.values())
            for sars, index in zip(data, [int(key) for key,_ in sorted(zip(keys,priorities), key = lambda item : item[1])]):

                self.at_file = index

                if self.keep_in_mem:
                    self.data[str(self.at_file)] = sars
                else:
                    self.data[str(self.at_file)] = "0"
                    with open( self.buffer_path + str(int(self.at_file/10000)) + "/" + str(self.at_file) + ".pkl", "wb" ) as f:
                        joblib.dump(sars,f)

                self.priorities[str(self.at_file)] = max(priorities)
                
        else:

            for sars in data:

                priorities = list(self.priorities.values())
                if self.keep_in_mem:
                    self.data[str(self.at_file)] = sars
                else:
                    self.data[str(self.at_file)] = "0"
                    with open( self.buffer_path + str(int(self.at_file/10000)) + "/" + str(self.at_file) + ".pkl", "wb" ) as f:
                        joblib.dump(sars,f)

                if len(priorities) == 0:
                    self.priorities[str(self.at_file)] = 1000
                else:
                    self.priorities[str(self.at_file)] = max(priorities)

                self.at_file += 1
                if self.at_file > self._max_buffer_size:
                    self.at_file = 0

    def get_indices(self, batch_size):

        if self.use_prioritized_replay:
            priorities = list(self.priorities.values())
            priorities = [priority**0.3 for priority in priorities]
            priorities_sum = (sum(priorities,start=0)+1)
            priorities = [priority/priorities_sum for priority in priorities]

            return rand.choices([int(key) for key in self.priorities.keys()], weights=priorities,k = batch_size)

        else: 
            return rand.choices([int(key) for key in self.priorities.keys()],k = batch_size)
    
    def sample_minibatch(self, indices):
        
        s_a_r_s = []

        for index in indices:

            if self.keep_in_mem:
                s_a_r_s.append(self.data[str(index)])
            else:
                with open( "./buffer/" + str(int(index/10000)) + "/" + str(index) + ".pkl", "rb" ) as f:
                    s_a_r_s.append(joblib.load(f))
        
        s = np.stack([elem[0] for elem in s_a_r_s],axis = 0)
        a = np.stack([elem[1] for elem in s_a_r_s],axis = 0)
        r = np.stack([elem[2] for elem in s_a_r_s],axis = 0)
        s_new = np.stack([elem[3] for elem in s_a_r_s],axis = 0)
        done = np.stack([elem[4] for elem in s_a_r_s],axis = 0)

        return {"values" : (s,a,r,s_new,done), "indices" : indices}


    def update_priority(self,priorities):

        for key, value in priorities.items():
            self.priorities[key] = value
        

class BufferManager(SyncManager):
    pass
