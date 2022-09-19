import gym
import random as rand
import tensorflow as tf
import numpy as np

import gym
import random as rand
import tensorflow as tf
import numpy as np
from gym.wrappers import AtariPreprocessing, FrameStack





def create_trajectory(model, batch : int, epsilon : float,env_name : str ,frame_skips : int,imgx : int, imgy : int):


    s_a_r_s = []

    # create action space
    ACTION_SPACE_SIZE = gym.make(env_name,full_action_space=False,new_step_api=True).action_space.n # only discrete

    #  create number of environments equal to batch size
    envs = [FrameStack(AtariPreprocessing(gym.make(env_name,full_action_space=False,new_step_api=True,frameskip=1), frame_skip=4, grayscale_obs = True, new_step_api=True,noop_max=0), num_stack=4, new_step_api=True) for _ in range(batch)]

    # create starting frame for each environment

    observations = [np.transpose(env.reset(),(1,2,0)) for env in envs]

    for _ in range(10000):
        

        if not len(observations) == len(envs):
            raise Exception("Length of lists 'observation' and 'envs' did not match")

        # create list that determines which actions will be sampled and which will be done with the model
        action_selector = [rand.randint(0,100)<=epsilon*100 for _ in range(len(envs))]

        # to avoid tf.function raytracing, ensure that always a tensor of the same shape is passed to the model
        while len(observations) < batch:
            observations.append(np.zeros(np.shape(observations[0])))


        # calculate the actions given the model using a batch
        with tf.device("/GPU:0"):
            actions = tf.argmax(model(tf.stack(observations,axis = 0),training = False), axis = -1).numpy()


        observations = observations[:len(envs)]
        actions = actions[:len(envs)]

        # sample new actions if the action selector says so
        actions = [rand.randint(0,ACTION_SPACE_SIZE-1) if should_sample_action else action for should_sample_action,action in zip(action_selector,actions)]

        result_from_env = []
        for env,action in zip(envs,actions):
            observation,reward,done,trunc,_ = env.step(action)
            reward = reward/10
            if trunc == True:
                done = True
            observation = np.transpose(observation,(1,2,0))
            result_from_env.append((observation,reward,done))

        done_and_remove_index = []
        for i, elem in enumerate(zip(observations,actions,result_from_env)):
            observation,action,(new_observation, reward, done) = elem

            s_a_r_s.append((observation.astype(np.uint8),action,reward,new_observation.astype(np.uint8), done))

            observations[i] = new_observation
            if done:
                done_and_remove_index.append(i)
            
        envs = [env for i,env in enumerate(envs) if not i in done_and_remove_index]
        observations = [observation for i,observation in enumerate(observations) if not i in done_and_remove_index]


        if len(envs) == 0:
            break
    
    return s_a_r_s


