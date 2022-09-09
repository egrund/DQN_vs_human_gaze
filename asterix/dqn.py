import threading
import numpy as np
import queue
import datetime
import gym
import socket
from sample_trajectory import create_trajectory
import time
import joblib
import pickle
import tensorflow as tf
import cProfile
import io 
import pstats

class agent:
    import tensorflow as tf
    def __init__(self, use_prioritized_replay : bool , env : str,epsilon : int,epsilon_decay : float, epsilon_min : float, batch_size : int,learning_rate :float,inner_its : int, samples_from_env : int,polyak_update : float, preload_weights_path : str):
        from model import AgentModel
        import tensorflow as tf
        self.inner_its = inner_its
        self.env = env
        self.batch_size = batch_size
        self.polyak_update = polyak_update
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        self.samples_from_env = samples_from_env
        self.epsilon_min = epsilon_min
        self.use_prioritized_replay = use_prioritized_replay

        
        self.model = AgentModel(gym.make(self.env,full_action_space=False,new_step_api=True).action_space.n)
        self.model_target = AgentModel(gym.make(self.env,full_action_space=False,new_step_api=True).action_space.n)

        # initialize weights 
        self.model(tf.random.uniform(shape=(1,84,84,12)))
        self.model_target(tf.random.uniform(shape=(1,84,84,12)))

        if preload_weights_path:
            self.model.load_weights(preload_weights_path + "model")
            self.model.load_weights(preload_weights_path + "model_target")
        else:
            self.model_target.set_weights(np.array(self.model.get_weights(),dtype = object))

    def train(self,its : int,path_model_weights : str,path_logs : str):
        import tensorflow as tf
        # https://www.tensorflow.org/tensorboard/get_started
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dqn_log_dir = path_logs + current_time + '/dqn'
        reward_log_dir = path_logs + current_time + '/reward'
        time_environment_sample_log_dir = path_logs + current_time + '/time_environment'
        time_step_log_dir = path_logs + current_time + '/time_step'

        dqn_summary_writer = tf.summary.create_file_writer(dqn_log_dir)
        reward_summary_writer = tf.summary.create_file_writer(reward_log_dir)
        time_environment_sample_summary_writer = tf.summary.create_file_writer(time_environment_sample_log_dir)
        time_step_summary_writer = tf.summary.create_file_writer(time_step_log_dir)

        current_epsilon = self.epsilon
        for i in range(its):

            # update epsilon
            if current_epsilon > self.epsilon_min:
                current_epsilon -= self.epsilon_decay


            ##################################################
            # sample training data from the buffer and sample new trajectory
            profiler = cProfile.Profile()
            profiler.enable()
            start_time_sample = time.time()   

            q_new_samples = queue.Queue()
            q_get_minibatch = queue.Queue()
            threads = []

            t_new_samples = threading.Thread(target = self.perform_sampling, args = (reward_summary_writer,current_epsilon,i,q_new_samples))  
            threads.append(t_new_samples)

            # preaload learning samples
            for k in range(self.inner_its):
                t = threading.Thread(target=self.get_minibatch, args = (8000+k, q_get_minibatch))
                threads.append(t)
            
            for t in threads:
                t.start()
            
            for t in threads:
                t.join()


            new_data = q_new_samples.get(block=True)
            training_data = []
            for _ in range(self.inner_its):
                training_data.append(q_get_minibatch.get(block=True))

            end_time_sample = time.time()

            profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
            ps.print_stats()
            with open('it_logs/it' + str(i) + '.txt', 'w+') as f:
                f.write(s.getvalue())

            #############################################################
            # add new data to the buffer and train

            start_time_step = time.time()


            t_training = threading.Thread(target=self.perform_training,args = (dqn_summary_writer,i, training_data) )
            t_update_buffer = threading.Thread(target = self.update_buffer, args = (new_data,))

            t_training.start()
            t_update_buffer.start()

            t_training.join()
            t_update_buffer.join()

            end_time_step = time.time()

            with time_step_summary_writer.as_default():
                tf.summary.scalar("time_step", end_time_step - start_time_step, step = i*self.inner_its)

            
            self.model.save_weights(path_model_weights + "model")
            self.model_target.save_weights(path_model_weights + "model_target")

            with open(path_model_weights + "optimizer.pkl", 'wb') as f:
                joblib.dump(self.optimizer.get_weights(), f)

            with time_environment_sample_summary_writer.as_default():
                tf.summary.scalar("time_environment", end_time_sample - start_time_sample, step = i*self.inner_its)
            
            print("round: ", i , " gpu memory: ", tf.config.experimental.get_memory_info("/GPU:0"))

    def update_buffer(self,data):
        soc = socket.socket()
        connected = False
        while not connected:
            try: 
                soc.connect(('localhost',7999))
                connected = True
            except:
                pass
        with soc:
            soc.sendall(pickle.dumps({"task" : "extend", "values" : data}))

    def get_minibatch(self, port, queue):
        fragments = []

        soc = socket.socket()
        connected = False
        while not connected:
            try: 
                soc.connect(('localhost',port))
                connected = True
            except:
                pass
        fragments = []
        with soc:
            while True:
                recvfile = soc.recv(4096)
                if not recvfile: break
                fragments.append(recvfile)
        queue.put(pickle.loads(b''.join(fragments)))
    
    def update_priorities(self,td):
        soc = socket.socket()
        connected = False
        while not connected:
            try: 
                soc.connect(('localhost',7999))
                connected = True
            except:
                pass
        with soc:
            soc.sendall(pickle.dumps({"task" : "priority", "values" : td}))

    def perform_sampling(self,reward_summary_writer,current_epsilon,i, q):
        
        # sample new trajectory
        new_data = create_trajectory(self.model,current_epsilon,self.env,4,84,84, self.samples_from_env)

        reward = []
        for _,_,r,_,_ in new_data:
            reward.append(tf.cast(r,tf.float32))

        # log average reward of average reward in tensorboard
        with reward_summary_writer.as_default():
            tf.summary.scalar("reward", tf.reduce_mean(reward), step = i*self.inner_its)
        
        
        q.put(new_data)

    def perform_training(self,dqn_summary_writer,i,data):

        
        for j in range(self.inner_its): 

            with tf.device("/CPU:0"):
                
                result = data[j]
                s,a,r,s_new,done = result["values"]
                indices = result["indices"]

                s = tf.cast(s,tf.float32)
                a = tf.cast(a,tf.float32)
                r = tf.cast(r,tf.float32)
                s_new = tf.cast(s_new,tf.float32)
                done = tf.cast(done,tf.float32)

            with tf.device("/GPU:0"):
                loss = self.model.step(s,a,r,s_new,done, self.optimizer, self.model_target)

                if self.use_prioritized_replay:
                    a_new = tf.argmax(self.model(s_new,training = False),axis = 1)
                    old_q_value = tf.gather(self.model(s,training=False),tf.cast(a,dtype=tf.int32),batch_dims=1)
                    new_q_value = tf.gather(self.model_target(s,training=False),a_new, batch_dims=1)

                    TD_error = r + tf.constant(0.99) * new_q_value - old_q_value
                    TD_error = TD_error.numpy()


            if self.use_prioritized_replay:
                td = {}
                for m,index in enumerate(indices):
                    td[str(index)] = TD_error[m]
                self.update_priorities(td)

            # apply polyak averaging
            self.model_target.set_weights((1-self.polyak_update)*np.array(self.model_target.get_weights(),dtype = object) + self.polyak_update*np.array(self.model.get_weights(),dtype = object))

            # log loss in tensorboard
            with dqn_summary_writer.as_default():
                tf.summary.scalar("dqn", loss, step=j+i*self.inner_its)

