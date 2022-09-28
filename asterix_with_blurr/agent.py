import numpy as np
import datetime
import gym
from blurr import to_blurred
from blurrer import Blurrer
from sample_trajectory import create_trajectory
import time
import joblib
import tensorflow as tf
from dqn import DQN


class agent:
    """
    This class implements the basic training procedure
    """

    def __init__(self, buffer, use_prioritized_replay : bool , env : str,epsilon : int,epsilon_decay : float, epsilon_decay_blurrer : float, start_blurrer_at : int,epsilon_min : float, batch_size : int,learning_rate :float,inner_its : int, samples_from_env : int,polyak_update : float, preload_weights_path : str):
        self.inner_its = inner_its
        self.env = env
        self.batch_size = batch_size
        self.polyak_update = polyak_update
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.optimizer_model = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        self.optimizer_blurrer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        self.samples_from_env = samples_from_env
        self.epsilon_min = epsilon_min
        self.use_prioritized_replay = use_prioritized_replay
        self.sample_correction = 1
        self.buffer = buffer
        self.epsilon_decay_blurrer = epsilon_decay_blurrer
        self.start_blurrer_at = start_blurrer_at

        
        self.model = DQN(gym.make(self.env,full_action_space=False,new_step_api=True).action_space.n)
        self.model_target = DQN(gym.make(self.env,full_action_space=False,new_step_api=True).action_space.n)

        self.blurrer = Blurrer()

        # initialize weights 
        self.model(tf.random.uniform(shape=(1,84,84,4)))
        self.model_target(tf.random.uniform(shape=(1,84,84,4)))
        self.blurrer(tf.random.uniform(shape=(1,84,84,4)))

        if preload_weights_path:
            self.model.load_weights(preload_weights_path + "model")
            self.model.load_weights(preload_weights_path + "model_target")
            self.blurrer.load_weights(preload_weights_path + "blurrer")
        else:
            self.model_target.set_weights(np.array(self.model.get_weights(),dtype = object))

    def train(self,its : int,path_model_weights : str,path_logs : str):

        # https://www.tensorflow.org/tensorboard/get_started
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dqn_log_dir = path_logs + current_time + '/dqn'
        reward_log_dir = path_logs + current_time + '/reward'
        time_environment_sample_log_dir = path_logs + current_time + '/time_environment'
        time_step_log_dir = path_logs + current_time + '/time_step'
        sample_correction_log_dir = path_logs + current_time + '/sample_correction'
        time_buffer_update_log_dir = path_logs + current_time + '/time_buffer_update'
        gaze_log_dir = path_logs + current_time + '/gaze'

        dqn_summary_writer = tf.summary.create_file_writer(dqn_log_dir)
        reward_summary_writer = tf.summary.create_file_writer(reward_log_dir)
        time_environment_sample_summary_writer = tf.summary.create_file_writer(time_environment_sample_log_dir)
        time_step_summary_writer = tf.summary.create_file_writer(time_step_log_dir)
        sample_correction_writer = tf.summary.create_file_writer(sample_correction_log_dir)
        time_buffer_update_writer = tf.summary.create_file_writer(time_buffer_update_log_dir)
        gaze_writer = tf.summary.create_file_writer(gaze_log_dir)

        current_epsilon = self.epsilon
        current_epsilon_blurrer = self.epsilon
        
        for i in range(its):

            # update epsilons
            if current_epsilon > self.epsilon_min:
                current_epsilon -= self.epsilon_decay
            if current_epsilon_blurrer > self.epsilon_min and i>self.start_blurrer_at:
                current_epsilon_blurrer -= self.epsilon_decay_blurrer

            ##################################################
            # train the dqn and blurrer 

            start_time_step = time.time()
            self.perform_training(dqn_summary_writer,gaze_writer,i)
            end_time_step = time.time()

            #############################################################
            # retrieve trajectory samples

            start_time_sample = time.time()
            new_data = self.perform_sampling(reward_summary_writer,sample_correction_writer,current_epsilon,current_epsilon_blurrer,i)
            end_time_sample = time.time()

            #############################################################
            # add new data to the buffer

            start_time_buffer = time.time()
            self.buffer.extend(new_data)
            end_time_buffer = time.time()

            with time_environment_sample_summary_writer.as_default():
                tf.summary.scalar("time_environment", end_time_sample - start_time_sample, step = i*self.inner_its)

            with time_step_summary_writer.as_default():
                tf.summary.scalar("time_step", end_time_step - start_time_step, step = i*self.inner_its)

            with time_buffer_update_writer.as_default():
                tf.summary.scalar("time_buffer_update", end_time_buffer - start_time_buffer, step = i*self.inner_its)

            self.model.save_weights(path_model_weights + "model")
            self.model_target.save_weights(path_model_weights + "model_target")
            self.blurrer.save_weights(path_model_weights + "blurrer")

            with open(path_model_weights + "optimizer.pkl", 'wb') as f:
                joblib.dump(self.optimizer_model.get_weights(), f)

            
            print("round: ", i , " gpu memory: ", tf.config.experimental.get_memory_info("/GPU:0"))

    def perform_sampling(self,reward_summary_writer,sample_correction_writer,current_epsilon,current_epsilon_blurrer,i):
        """
        This function performs the basic sampling given the current state of the model and epsilon
        Args:  
            - reward_summary_writer <tf.summary.SummaryWriter>: Sumary writer for the reward
            - sample_correlation_writer <tf.summary.SummaryWriter>: Summary writer for the sample correlation
            - current_epsilon <float>: current epsilon for the dqn
            - current_epsilon_blurrer <float> : current epsilon for the Blurrer
            - i <int> : current iteration
        Returns:
            - List<tuple> : new data sampked from environment
        """
        
        # sample new trajectory
        new_data = create_trajectory(self.model,self.blurrer,int(18*self.sample_correction)+1, current_epsilon,current_epsilon_blurrer,self.env)

        reward = []
        for _,_,r,_,_,_,_,_,_ in new_data:
            reward.append(tf.cast(r,tf.float32))

        # log average reward of average reward in tensorboard
        with reward_summary_writer.as_default():
            tf.summary.scalar("reward", 10*tf.reduce_sum(reward)/(int(18*self.sample_correction)+1), step = i*self.inner_its)

        with sample_correction_writer.as_default():
            tf.summary.scalar("sample_correction", self.sample_correction, step = i*self.inner_its)

        self.sample_correction = (self.samples_from_env/len(new_data))*self.sample_correction

        
        return new_data
    
    @tf.function
    def calculate_td_error(self, s, a, r,s_new):
        """
        Calculates the td error for datapoints s,a,r,s'
        Args:
            - s <tf.Tensor> : state s from the environment
            - a <tf.Tensor> : action a performed in s
            - r <tf.Tensor> : reward received from performing a in s
            - s_new <tf.Tensor> : new state after performing a in s
        Returns:
            - <tf.Tensor> : tf-error given the current dqn and dqn_target state
        """

        a_new = tf.argmax(self.model(s_new,training = False),axis = 1)
        old_q_value = tf.gather(self.model(s,training=False),tf.cast(a,dtype=tf.int32),batch_dims=1)
        new_q_value = tf.gather(self.model_target(s,training=False),a_new, batch_dims=1)

        TD_error = tf.abs(r + tf.constant(0.99) * new_q_value - old_q_value)
        return TD_error

    def perform_training(self,dqn_summary_writer,gaze_writer,i):
        """
        Perform a training step
        Args:
            - dqn_summary_writer <tf.summary.SummaryWriter> : Summary writer for the dqn reward
            - gaze_writer <tf.summary.SummaryWriter> : Summary writer for the gaze positions
            - i <int> : current iteration
        Returns:
            
        """

       
        for j in range(self.inner_its): 
            td = {}

            with tf.device("/CPU:0"):
                
                training_data = self.buffer.sample_minibatch(self.buffer.get_indices(self.batch_size))
                s,a,r,s_new,done,x_s,y_s,x_s_new,y_s_new = training_data["values"]
                indices = training_data["indices"]

                s = tf.cast(s,tf.float32)
                a = tf.cast(a,tf.float32)
                r = tf.cast(r,tf.float32)
                done = tf.cast(done,tf.float32)
                s_new = tf.cast(s_new,tf.float32)
                x_s = tf.cast(x_s, tf.float32)
                y_s = tf.cast(y_s, tf.float32)
                x_s_new = tf.cast(x_s_new, tf.float32)
                y_s_new = tf.cast(y_s_new, tf.float32)

                s_new_unblurred = s_new
                

            with tf.device("/GPU:0"):

                s = to_blurred(s,x_s,y_s,4)
                s_new = to_blurred(s_new,x_s_new,y_s_new,4)

                loss_model = self.model.step(s,a,r,s_new,done,self.optimizer_model, self.model_target)

                if self.use_prioritized_replay:
                    TD_error = self.calculate_td_error(s,a,r,s_new).numpy()
            
            
            if self.use_prioritized_replay:
                for m,index in enumerate(indices):
                    td[str(index)] = TD_error[m]
                self.buffer.update_priorities(td)

            # log loss in tensorboard
            if dqn_summary_writer:
                with dqn_summary_writer.as_default():
                    tf.summary.scalar("dqn", loss_model, step=j+i*self.inner_its)
            
            with tf.device("/CPU:0"):
                
                training_data = self.buffer.sample_minibatch(self.buffer.get_indices(32))
                s,a,r,s_new,done,x_s,y_s,x_s_new,y_s_new = training_data["values"]
                indices = training_data["indices"]

                s = tf.cast(s,tf.float32)
                a = tf.cast(a,tf.float32)
                r = tf.cast(r,tf.float32)
                done = tf.cast(done,tf.float32)
                s_new = tf.cast(s_new,tf.float32)
                x_s = tf.cast(x_s, tf.float32)
                y_s = tf.cast(y_s, tf.float32)
                x_s_new = tf.cast(x_s_new, tf.float32)
                y_s_new = tf.cast(y_s_new, tf.float32)

            with tf.device("/GPU:0"):
                if i>self.start_blurrer_at:
                    s_new_unblurred = s_new
                    s = to_blurred(s,x_s,y_s,4)
                    loss_blurrer,x,y = self.blurrer.step(s,s_new_unblurred,self.optimizer_blurrer,self.model)
                else:
                    loss_blurrer = 0
                    x = 0
                    y = 0

            # log loss in tensorboard
            if dqn_summary_writer:
                with dqn_summary_writer.as_default():
                    
                    tf.summary.scalar("blurrer", loss_blurrer, step=j+i*self.inner_its)
            if gaze_writer:
                if i>self.start_blurrer_at:
                    with gaze_writer.as_default():
                        tf.summary.scalar("x_0", tf.reduce_mean(x.numpy()[:,0]), step=j+i*self.inner_its)
                        tf.summary.scalar("y_0", tf.reduce_mean(y.numpy()[:,0]), step=j+i*self.inner_its)
                        tf.summary.scalar("x_1", tf.reduce_mean(x.numpy()[:,1]), step=j+i*self.inner_its)
                        tf.summary.scalar("y_1", tf.reduce_mean(y.numpy()[:,1]), step=j+i*self.inner_its)
                        tf.summary.scalar("x_2", tf.reduce_mean(x.numpy()[:,2]), step=j+i*self.inner_its)
                        tf.summary.scalar("y_2", tf.reduce_mean(y.numpy()[:,2]), step=j+i*self.inner_its)
                        tf.summary.scalar("x_3", tf.reduce_mean(x.numpy()[:,3]), step=j+i*self.inner_its)
                        tf.summary.scalar("y_3", tf.reduce_mean(y.numpy()[:,3]), step=j+i*self.inner_its)

            

        # apply polyak averaging
        self.model_target.set_weights((1-self.polyak_update)*np.array(self.model_target.get_weights(),dtype = object) + self.polyak_update*np.array(self.model.get_weights(),dtype = object))


