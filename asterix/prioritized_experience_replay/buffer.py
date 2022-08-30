import tensorflow as tf
import random as rand
import threading, queue
import numpy as np
import gym

class Buffer:

    def __init__(self, max_buffer_size, min_buffer_size) -> None:
        self._data = []
        self._max_buffer_size = max_buffer_size
        self._min_buffer_size = min_buffer_size
        self.last_batch_indexes = list()

    def extend(self, list):
        list_p = []
        for e in list:
            list_p.append(e + (np.inf,)) # give max priority
        self._data.extend(list_p) 
        # heapsort 
        self._data.sort(reverse=True,key=self.priority)
        # remove old elements if buffer overflows
        while len(self._data)>self._max_buffer_size:
            self._data = self._data[1:]


    def fill(self, num_threads, function, model, epsilon,env):
        q = queue.Queue()
        threads = []
        while len(self._data)<self._min_buffer_size:

            while len(threads) < num_threads:
                environment = gym.make(env,full_action_space=False,new_step_api=True)
                thread = threading.Thread(target=function, args=(q,model,False,epsilon,environment))
                thread.start()
                threads.append(thread)

            threads = [t for t in threads if thread.is_alive()]

            while not q.empty():
                elem = q.get(block=False)
                elem_p = []
                for e in elem:
                    elem_p.append(e + (np.inf,)) # give max priority
                self._data.extend(elem_p)
                print("Filling buffer: ", len(self._data), "/", self._min_buffer_size)

        # heapsort
        self._data.sort(reverse=True,key=self.priority)


    def sample_minibatch(self, batch_size):
        """
        return a minibatch sampled from the buffer
        """
        self.last_batch_indexes = [0]*batch_size
        s_batch = tf.TensorArray(tf.float32,size = batch_size)
        a_batch = tf.TensorArray(tf.float32,size = batch_size)
        r_batch = tf.TensorArray(tf.float32,size = batch_size)
        s_new_batch = tf.TensorArray(tf.float32,size = batch_size)
        done_batch = tf.TensorArray(tf.float32,size = batch_size)
        for i in range(batch_size):
            # element = self._data[i] # take first elements from heap as greedy choice
            # stochastical rank-based prioritization
            indexes  = np.arange(0,len(self._data))
            probs = 1/np.arange(len(self._data),0,-1)
            index = rand.choices(indexes, weights = probs, k=1 )[0]
            self.last_batch_indexes[i] = index
            s,a,r,s_new,done,_p = self._data[index]

            # cast all elements to floats
            r = tf.cast(r,tf.float32)
            s = tf.cast(s,tf.float32)
            a = tf.cast(a,tf.float32)
            s_new = tf.cast(s_new,tf.float32)
            done = tf.cast(tf.cast(done,tf.int32),tf.float32)

            # add them to the tensor array
            s_batch = s_batch.write(i,s)
            a_batch = a_batch.write(i,a)
            r_batch = r_batch.write(i,r)    
            s_new_batch = s_new_batch.write(i,s_new)
            done_batch = done_batch.write(i,done)
        
        # stack for batch dimension
        return s_batch.stack(),a_batch.stack(),r_batch.stack(),s_new_batch.stack(),done_batch.stack()

    def priority(self,elem):
        """ returns the priority element for one data sample """
        return elem[-1]

    def update_priority(self,priorities):
        """ updates the priorities of the last sampled minibatch """

        if len(priorities) != len(self.last_batch_indexes):
            print("Error, last batch size was not the same as update size is.")
            return

        for i,s in enumerate(self.last_batch_indexes):
            sample = list(self._data[s])
            sample[-1] = priorities[i]
            self._data[s] = tuple(sample)

        # heapsort
        self._data.sort(reverse=True,key=self.priority)
