import tensorflow as tf
import numpy as np
import datetime
import gym
from buffer import Buffer
from sample_trajectory import create_trajectory_thread,create_trajectory

class model(tf.keras.Model):
    def __init__(self, num_actions = 9.0, input_shape = (84,84,4)):
        super(model, self).__init__()

        self._mse = tf.keras.losses.MeanSquaredError()

        self._l1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
        self._l2 = tf.keras.layers.MaxPooling2D((2, 2))
        self._l3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self._l4 = tf.keras.layers.MaxPooling2D((2, 2))
        self._l5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self._l6 = tf.keras.layers.GlobalMaxPooling2D()
        self._l7 = tf.keras.layers.Dense(64,activation="relu")
        self._l8 = tf.keras.layers.Dense(32,activation="relu")
        self._l9 = tf.keras.layers.Dense(16,activation="relu")

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
        x = self._l8(x,training=training)
        x = self._l9(x,training=training)

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

    def __init__(self,env = gym.make("ALE/Asterix-v5",full_action_space=False,new_step_api=True),epsilon=1,epsilon_decay=0.09,batch_size = 512,optimizer = tf.keras.optimizers.Adam(0.00025, beta_1=0.9, beta_2=0.999, epsilon=1e-07),inner_its=80,polyak_update = 0.025,buffer_size=40000, buffer_min=38000,threads=10):
        self.inner_its = 80
        self.threads = 10
        self.env = env
        self.batch_size = batch_size
        self.polyak_update = polyak_update
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.optimizer = optimizer
        
        self.model = model(env.action_space.n)
        self.model_target = model(env.action_space.n)
        self.buffer = Buffer(buffer_size, buffer_min)

        # initialize weights 
        self.model(tf.random.uniform(shape=(1,84,84,4)))
        self.model_target(tf.random.uniform(shape=(1,84,84,4)))
        self.model_target.set_weights(np.array(self.model.get_weights(),dtype = object))
        # initialize buffer
        self.buffer.fill(self.threads,create_trajectory_thread,self.model,1,self.env)

    def train(self,its=20000,path ='logs/asterix_test/run1'):

        current_epsilon = self.epsilon

        # https://www.tensorflow.org/tensorboard/get_started
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dqn_log_dir = path + current_time + '/dqn'
        reward_log_dir = path + current_time + '/reward'
        dqn_summary_writer = tf.summary.create_file_writer(dqn_log_dir)
        reward_summary_writer = tf.summary.create_file_writer(reward_log_dir)

        for i in range(its):

            # apply polyak averaging
            self.model_target.set_weights((1-self.polyak_update)*np.array(self.model_target.get_weights(),dtype = object) + self.polyak_update*np.array(self.model.get_weights(),dtype = object))

            # sample new trajectory
            new_data = create_trajectory(self.model,False,current_epsilon,self.env)
            if current_epsilon > 0.1:
                current_epsilon -= self.epsilon_decay
            reward = []
            for s,a,r,new_s,done in new_data:
                reward.append(tf.cast(r,tf.float32))
            #print("round: ", i," average reward: ",tf.reduce_mean(reward))

            # log average reward in tensorboard
            with reward_summary_writer.as_default():
                tf.summary.scalar("reward", tf.reduce_mean(reward), step = i*self.inner_its)

            # add new data to replay buffer
            self.buffer.extend(new_data)

            for j in range(self.inner_its):
                s,a,r,s_new,done  = self.buffer.sample_minibatch(self.batch_size)

                loss = self.model.step(s,a,r,s_new,done, self.optimizer, self.model_target)
                TD_error = list()
                for i in range(a.shape[0]):
                    a_new = tf.argmax(tf.squeeze(self.model(tf.expand_dims(s_new[i],axis=0),training = False),axis = 0)).numpy()
                    old_q_value = tf.squeeze(self.model(tf.expand_dims(s[i],axis=0),training=False), axis = 0)[tf.cast(a[i],dtype=tf.int32)]
                    new_q_value = tf.squeeze(self.model_target(tf.expand_dims(s[i],axis=0), training = False), axis = 0)[a_new]
                    TD_error.append( (r[i] + 0.99 * new_q_value - old_q_value).numpy())
                self.buffer.update_priority(TD_error)

                # log loss in tensorboard
                with dqn_summary_writer.as_default():
                    tf.summary.scalar("dqn", loss, step=j+i*self.inner_its)

            self.model.save_weights(path)
            self.model_target.save_weights(path)