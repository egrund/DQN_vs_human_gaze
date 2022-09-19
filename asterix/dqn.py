import numpy as np
import datetime
import gym
from sample_trajectory import create_trajectory
import time
import joblib
import tensorflow as tf
class agent:
    import tensorflow as tf
    def __init__(self, buffer, use_prioritized_replay : bool , env : str,epsilon : int,epsilon_decay : float, epsilon_min : float, batch_size : int,learning_rate :float,inner_its : int, samples_from_env : int,polyak_update : float, preload_weights_path : str):
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
        self.sample_correction = 1
        self.buffer = buffer

        
        self.model = AgentModel(gym.make(self.env,full_action_space=False,new_step_api=True).action_space.n)
        self.model_target = AgentModel(gym.make(self.env,full_action_space=False,new_step_api=True).action_space.n)

        # initialize weights 
        self.model(tf.random.uniform(shape=(1,84,84,4)))
        self.model_target(tf.random.uniform(shape=(1,84,84,4)))

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
        sample_correction_log_dir = path_logs + current_time + '/sample_correction'
        time_buffer_update_log_dir = path_logs + current_time + '/time_buffer_update'

        dqn_summary_writer = tf.summary.create_file_writer(dqn_log_dir)
        reward_summary_writer = tf.summary.create_file_writer(reward_log_dir)
        time_environment_sample_summary_writer = tf.summary.create_file_writer(time_environment_sample_log_dir)
        time_step_summary_writer = tf.summary.create_file_writer(time_step_log_dir)
        sample_correction_writer = tf.summary.create_file_writer(sample_correction_log_dir)
        time_buffer_update_writer = tf.summary.create_file_writer(time_buffer_update_log_dir)

        current_epsilon = self.epsilon
        
        for i in range(its):
            #print("buffer size in gb: ", asizeof.asizeof(self.buffer)/1000000000)
            #print("agent size in gb: ", asizeof.asizeof(self)/1000000000)
            # update epsilon
            if current_epsilon > self.epsilon_min:
                current_epsilon -= self.epsilon_decay

            ##################################################
            # train 
            #memory_tracker = tracker.SummaryTracker()
            start_time_step = time.time()
            #profiler = cProfile.Profile()
            #profiler.enable()
            self.perform_training(dqn_summary_writer,i)
            
            #profiler.disable()
            #s = io.StringIO()
            #ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
            #ps.print_stats()
            #with open('it_logs/it' + str(i) + 'training.txt', 'w+') as f:
            #    f.write(s.getvalue())
            end_time_step = time.time()

            #memory_tracker.print_diff()

            #############################################################
            # retrieve trajectory samples

            start_time_sample = time.time()
            #profiler = cProfile.Profile()
            #profiler.enable()
            new_data = self.perform_sampling(reward_summary_writer,sample_correction_writer,current_epsilon,i)
            #profiler.disable()
            #s = io.StringIO()
            #ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
            #ps.print_stats()
            #with open('it_logs/it' + str(i) + 'sampling.txt', 'w+') as f:
            #    f.write(s.getvalue())

            end_time_sample = time.time()

            #############################################################
            # add new data to the buffer

            #memory_tracker.print_diff()

            start_time_buffer = time.time()
            self.buffer.extend(new_data)
            end_time_buffer = time.time()

            #memory_tracker.print_diff()

            with time_environment_sample_summary_writer.as_default():
                tf.summary.scalar("time_environment", end_time_sample - start_time_sample, step = i*self.inner_its)

            with time_step_summary_writer.as_default():
                tf.summary.scalar("time_step", end_time_step - start_time_step, step = i*self.inner_its)

            with time_buffer_update_writer.as_default():
                tf.summary.scalar("time_buffer_update", end_time_buffer - start_time_buffer, step = i*self.inner_its)

            self.model.save_weights(path_model_weights + "model")
            self.model_target.save_weights(path_model_weights + "model_target")

            with open(path_model_weights + "optimizer.pkl", 'wb') as f:
                joblib.dump(self.optimizer.get_weights(), f)

            
            print("round: ", i , " gpu memory: ", tf.config.experimental.get_memory_info("/GPU:0"))

    def perform_sampling(self,reward_summary_writer,sample_correction_writer,current_epsilon,i):
        
        # sample new trajectory
        new_data = create_trajectory(self.model,int(18*self.sample_correction)+1, current_epsilon,self.env, 4,84,84)

        self.sample_correction = (self.samples_from_env/len(new_data))*self.sample_correction

        reward = []
        for _,_,r,_,_ in new_data:
            reward.append(tf.cast(r,tf.float32))

        # log average reward of average reward in tensorboard
        with reward_summary_writer.as_default():
            tf.summary.scalar("reward", tf.reduce_sum(reward), step = i*self.inner_its)

        with sample_correction_writer.as_default():
            tf.summary.scalar("sample_correction", self.sample_correction, step = i*self.inner_its)
    
        
        return new_data
    
    @tf.function
    def calculate_td_error(self, s, a, r,s_new):

        a_new = tf.argmax(self.model(s_new,training = False),axis = 1)
        old_q_value = tf.gather(self.model(s,training=False),tf.cast(a,dtype=tf.int32),batch_dims=1)
        new_q_value = tf.gather(self.model_target(s,training=False),a_new, batch_dims=1)

        TD_error = tf.abs(r + tf.constant(0.99) * new_q_value - old_q_value)
        return TD_error

    def perform_training(self,dqn_summary_writer,i):



       
        for j in range(self.inner_its): 
            td = {}

            with tf.device("/CPU:0"):
                
                training_data = self.buffer.sample_minibatch(self.buffer.get_indices(self.batch_size))
                s,a,r,s_new,done = training_data["values"]
                indices = training_data["indices"]

                a = tf.cast(a,tf.float32)
                r = tf.cast(r,tf.float32)
                done = tf.cast(done,tf.float32)

            with tf.device("/GPU:0"):
                loss = self.model.step(s,a,r,s_new,done, self.optimizer, self.model_target)

                if self.use_prioritized_replay:
                    TD_error = self.calculate_td_error(s,a,r,s_new).numpy()


            if self.use_prioritized_replay:
                for m,index in enumerate(indices):
                    td[str(index)] = TD_error[m]
                self.buffer.update_priorities(td)

        # apply polyak averaging
        self.model_target.set_weights((1-self.polyak_update)*np.array(self.model_target.get_weights(),dtype = object) + self.polyak_update*np.array(self.model.get_weights(),dtype = object))

        # log loss in tensorboard
        if dqn_summary_writer:
            with dqn_summary_writer.as_default():
                tf.summary.scalar("dqn", loss, step=j+i*self.inner_its)

