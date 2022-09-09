from multiprocessing.managers import SyncManager
import joblib
import os
import random as rand
import numpy as np

class Buffer:

    def __init__(self, max_buffer_size, min_buffer_size, buffer_path, keep_in_mem, use_prioritized_replay) -> None:

        self.keep_in_mem = keep_in_mem
        self.buffer_path = buffer_path

        self.at_file = 0

        self._max_buffer_size = max_buffer_size
        self._min_buffer_size = min_buffer_size
        self.use_prioritized_replay = use_prioritized_replay
        self.last_batch_indexes = list()
        
        self.priorities = {}
        self.data = {}



    def clear(self):

        if self.keep_in_mem:
            self.data = {}
        else:
            for (dirpath, _, filenames) in os.walk(self.buffer_path):
                for filename in filenames:
                    os.remove(os.path.join(dirpath,filename))
                break
        self.priorities = {}

    def extend(self, data):

        if self.use_prioritized_replay and not len(self.data.keys()) < self._max_buffer_size:
            self.at_file = rand.randint(0,len(self.data.keys())-1)

            for sars in data:

                if self.keep_in_mem:
                    self.data[str(self.at_file)] = sars
                else:
                    self.data[str(self.at_file)] = "0"
                    with open( self.buffer_path +  str(self.at_file) + ".pkl", "wb" ) as f:
                        joblib.dump(sars,f)

                self.priorities[str(self.at_file)] = np.inf
                
        else:

            for sars in data:

                if self.keep_in_mem:
                    self.data[str(self.at_file)] = sars
                else:
                    self.data[str(self.at_file)] = "0"
                    with open( self.buffer_path +  str(self.at_file) + ".pkl", "wb" ) as f:
                        joblib.dump(sars,f)

                self.priorities[str(self.at_file)] = np.inf

                self.at_file += 1
                if self.at_file > self._max_buffer_size:
                    self.at_file = 0

    
    def sample_minibatch(self, batch_size):

        if self.use_prioritized_replay:
            pass
            indices = rand.choices([int(key) for key in self.data.keys()],weights = self.priorities.values(), k = batch_size)

            for index in indices:
                self.priorities[str(index)] = 0

        else: 
            indices = rand.choices([int(key) for key in self.data.keys()],k = batch_size)
        
        s_a_r_s = []

        for index in indices:

            if self.keep_in_mem:
                s_a_r_s.append(self.data[str(index)])
            else:
                with open( "./buffer/" +  str(index) + ".pkl", "rb" ) as f:
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