import random as rand
import numpy as np
import joblib
from sample_trajectory import create_trajectory



class Buffer:

    def __init__(self, max_buffer_size, min_buffer_size) -> None:

        self._data_s = []
        self._data_a = []
        self._data_r = []
        self._data_s_new = []
        self._data_done = []
        self._data_priority = []

        self._max_buffer_size = max_buffer_size
        self._min_buffer_size = min_buffer_size
        self.last_batch_indexes = list()

    def sort(self):

        # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        argsort = sorted(range(len(self._data_priority)), key=self._data_priority.__getitem__)
        argsort.reverse()

        self._data_s = [self._data_s[i] for i in argsort]
        self._data_a = [self._data_a[i] for i in argsort]
        self._data_r = [self._data_r[i] for i in argsort]
        self._data_s_new = [self._data_s_new[i] for i in argsort]
        self._data_done = [self._data_done[i] for i in argsort]
        self._data_priority = [self._data_priority[i] for i in argsort]
    
    def reset(self):
        self._data_s = []
        self._data_a = []
        self._data_r = []
        self._data_s_new = []
        self._data_done = []
        self._data_priority = []

    def load(self, path : str):

        try:
            with open( path + "bufferv3_s.pkl", "rb" ) as f:
                self._data_s = joblib.load(f)
            with open( path + "bufferv3_a.pkl", "rb" ) as f:
                self._data_a = joblib.load(f)
            with open( path + "bufferv3_r.pkl", "rb" ) as f:
                self._data_r = joblib.load(f)
            with open( path + "bufferv3_s_new.pkl", "rb" ) as f:
                self._data_s_new = joblib.load(f)
            with open( path + "bufferv3_done.pkl", "rb" ) as f:
                self._data_done = joblib.load(f)
            with open( path + "bufferv3_priority.pkl", "rb" ) as f:
                self._data_priority = joblib.load(f)
        except:
            print("Buffer could not be loaded")
            self.reset()
    
    def save(self, path : str):
        try:
            with open( path + "bufferv4_s.pkl", "wb" ) as f:
                joblib.dump(self._data_s, f)
            with open( path + "bufferv4_a.pkl", "wb" ) as f:
                joblib.dump(self._data_a, f)
            with open( path + "bufferv4_r.pkl", "wb" ) as f:
                joblib.dump(self._data_r, f)
            with open( path + "bufferv4_s_new.pkl", "wb" ) as f:
                joblib.dump(self._data_s_new, f)
            with open( path + "bufferv4_done.pkl", "wb" ) as f:
                joblib.dump(self._data_done, f)
            with open( "bufferv4_priority.pkl", "wb" ) as f:
                joblib.dump(self._data_priority, f)
        except:
            print("Buffer could not be saved")

    def extend(self, data):

        for s,a,r,s_new,done in data:

            self._data_s.insert(0,s)
            self._data_a.insert(0,a)
            self._data_r.insert(0,r)
            self._data_s_new.insert(0,s_new)
            self._data_done.insert(0,done)
            self._data_priority.insert(0,np.inf)

        
        # remove old elements if buffer overflows
        self._data_s = self._data_s[:self._max_buffer_size]
        self._data_a = self._data_a[:self._max_buffer_size]
        self._data_r = self._data_r[:self._max_buffer_size]
        self._data_s_new = self._data_s_new[:self._max_buffer_size]
        self._data_done = self._data_done[:self._max_buffer_size]
        self._data_priority = self._data_priority[:self._max_buffer_size]


    def fill(self, model, epsilon,env):

       
        while len(self._data_s)<self._min_buffer_size:
            data = create_trajectory(model,100,epsilon,env,4,84,84)
            self.extend(data)
            print("Filling buffer: ", len(self._data_s), "/", self._min_buffer_size)
        

    def sample_minibatch(self, batch_size, use_prioritized_replay):
        """
        return a minibatch sampled from the buffer
        """
        if use_prioritized_replay:
            indices = rand.choices(range(len(self._data_s)),weights = self._data_priority, k = batch_size)
        else: 
            indices = rand.choices(range(len(self._data_s)),k = batch_size)

        s = np.stack([self._data_s[i] for i in indices],axis = 0)
        a = np.stack([self._data_a[i] for i in indices],axis = 0)
        r = np.stack([self._data_r[i] for i in indices],axis = 0)
        s_new = np.stack([self._data_s_new[i] for i in indices],axis = 0)
        done = np.stack([self._data_done[i] for i in indices],axis = 0)


        return s,a,r,s_new,done

    def update_priority(self,priorities):
        """ updates the priorities of the last sampled minibatch """

        if len(priorities) != len(self.last_batch_indexes):
            print("Error, last batch size was not the same as update size is.")
            return

        for i,s in enumerate(self.last_batch_indexes):
            self._data_priority[s] = priorities[i]

        # heapsort
        self.sort()
