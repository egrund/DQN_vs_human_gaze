import imp
import tensorflow as tf
import numpy as np
import datetime
import gym
from buffer import Buffer
import pickle
from sample_trajectory import create_trajectory_thread,create_trajectory
import tqdm
from numba import cuda
import joblib

class model(tf.keras.Model):
    def __init__(self, num_actions = 9.0, input_shape = (84,84,4)):
        super(model, self).__init__()

        self._mse = tf.keras.losses.MeanSquaredError()

        self._l1 = tf.keras.layers.Conv2D(64, (8, 8), strides = (2,2), activation='relu', input_shape=input_shape)#, kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2=0.1))
        self._l2 = tf.keras.layers.MaxPooling2D((3, 3))
        self._l3 = tf.keras.layers.Conv2D(64, (4, 4), activation='relu')#, kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2=0.1))
        self._l4 = tf.keras.layers.MaxPooling2D((3, 3))
        self._l5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')#, kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2=0.1))
        self._l6 = tf.keras.layers.GlobalMaxPooling2D()
        self._l7 = tf.keras.layers.Dense(512,activation="relu")#, kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2=0.1))

        # state value
        self._l10 = tf.keras.layers.Dense(1,activation="linear")

        # advantages
        self._num_actions = num_actions
        self._l11 = tf.keras.layers.Dense(self._num_actions,activation="linear")

    @tf.function
    def call(self,x ,training):
        x = self._l1(x,training=training)
        x = self._l2(x,training=training)
        x = self._l3(x,training=training)
        x = self._l4(x,training=training)
        x = self._l5(x,training=training)
        x = self._l6(x,training=training)
        x = self._l7(x,training=training)
        #x = self._l8(x,training=training)
        #x = self._l9(x,training=training)

        state_value = self._l10(x,training=training)
        advantages = self._l11(x,training=training)

        return state_value + advantages - tf.expand_dims(tf.reduce_sum(advantages,axis = -1)/tf.cast(self._num_actions,tf.float32),axis = -1)
    
    @tf.function
    def step(self,s,a,r,s_new,done,optimizer,model_target):
        with tf.GradientTape() as tape:

            with tf.device("/GPU:0"):

                # calculate the corresponding q values
                Q_max = tf.math.reduce_max(model_target(s_new),axis=1)
                Q_s_a = tf.gather(params = self(s),indices = tf.cast(a,tf.int32),axis = 1,batch_dims = 1)

                # apply mean squared error loss
                loss = self._mse(Q_s_a, r + (tf.constant(0.99)*Q_max)*(1-done))

        # perform gradient descent step
        grads = tape.gradient(loss, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss



class agent:

    def __init__(self,buffer : Buffer, use_prioritized_replay : bool , env : str,epsilon : int,epsilon_decay : float, epsilon_min : float, batch_size : int,optimizer : tf.keras.optimizers,inner_its : int, samples_from_env : int,polyak_update : float):

        self.inner_its = inner_its
        self.env = gym.make(env,full_action_space=False,new_step_api=True)
        self.batch_size = batch_size
        self.polyak_update = polyak_update
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.optimizer = optimizer
        self.buffer = buffer
        self.samples_from_env = samples_from_env
        self.epsilon_min = epsilon_min
        self.use_prioritized_replay = use_prioritized_replay

        
        self.model = model(self.env.action_space.n)
        self.model_target = model(self.env.action_space.n)

        # initialize weights 
        self.model(tf.random.uniform(shape=(1,84,84,4)))
        self.model_target(tf.random.uniform(shape=(1,84,84,4)))
        self.model_target.set_weights(np.array(self.model.get_weights(),dtype = object))

    def train(self,its : int,path_model_weights : str,path_logs : str):

        current_epsilon = self.epsilon

        # https://www.tensorflow.org/tensorboard/get_started
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dqn_log_dir = path_logs + current_time + '/dqn'
        reward_log_dir = path_logs + current_time + '/reward'
        dqn_summary_writer = tf.summary.create_file_writer(dqn_log_dir)
        reward_summary_writer = tf.summary.create_file_writer(reward_log_dir)

        for i in range(its):

            # update epsilon
            if self.epsilon_min > 0.1:
                current_epsilon -= self.epsilon_decay

            
            for m in range(self.samples_from_env):
                # sample new trajectory
                new_data, _performance = create_trajectory(self.model,False,current_epsilon,self.env)

                reward = []
                for s,a,r,new_s,done in new_data:
                    reward.append(tf.cast(r,tf.float32))
                print("round: ", i," average reward: ",tf.reduce_mean(reward))

                # log average reward in tensorboard
                with reward_summary_writer.as_default():
                    tf.summary.scalar("reward", tf.reduce_mean(reward), step = m+i*self.inner_its)

                # add new data to replay buffer
                self.buffer.extend(new_data)

        
            for j in range(self.inner_its):

                s,a,r,s_new,done  = self.buffer.sample_minibatch(self.batch_size, self.use_prioritized_replay)
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
                    self.buffer.update_priority(TD_error)

                # apply polyak averaging
                self.model_target.set_weights((1-self.polyak_update)*np.array(self.model_target.get_weights(),dtype = object) + self.polyak_update*np.array(self.model.get_weights(),dtype = object))

                # log loss in tensorboard
                with dqn_summary_writer.as_default():
                    tf.summary.scalar("dqn", loss, step=j+i*self.inner_its)


            self.model.save_weights(path_model_weights)
            self.model_target.save_weights(path_model_weights)




    def evaluate(self,its=10):
        performance = 0
        for _ in range(its):
            _,p = create_trajectory(self.model)
            performance += p
        av_performance = performance / its
        return av_performance
