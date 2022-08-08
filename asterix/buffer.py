import tensorflow as tf
import random as rand
import threading, queue


class Buffer:

    def __init__(self, max_buffer_size, min_buffer_size) -> None:
        self._data = []
        self._max_buffer_size = max_buffer_size
        self._min_buffer_size = min_buffer_size

    def extend(self, list):
        self._data.extend(list)

        # remove old elements if buffer overflows
        while len(self._data)>self._max_buffer_size:
            self._data = self._data[1:]


    def fill(self, num_threads, function, model, epsilon,env):
        q = queue.Queue()
        threads = []
        while len(self._data)<self._min_buffer_size:

            while len(threads) < num_threads:
                thread = threading.Thread(target=function, args=(q,model,False,epsilon,env))
                thread.start()
                threads.append(thread)

            threads = [t for t in threads if thread.is_alive()]

            while not q.empty():
                elem = q.get(block=False)
                self._data.extend(elem)
                print("Filling buffer: ", len(self._data), "/", self._min_buffer_size)


    def sample_minibatch(self, batch_size):
        """
        return a minibatch sampled from the buffer
        """
        s_batch = tf.TensorArray(tf.float32,size = batch_size)
        a_batch = tf.TensorArray(tf.float32,size = batch_size)
        r_batch = tf.TensorArray(tf.float32,size = batch_size)
        s_new_batch = tf.TensorArray(tf.float32,size = batch_size)
        done_batch = tf.TensorArray(tf.float32,size = batch_size)
        for i in range(batch_size):
            element = rand.choice(self._data)
            s,a,r,s_new,done = element

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