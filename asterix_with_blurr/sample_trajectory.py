import gym
import random as rand
import tensorflow as tf
import numpy as np

import gym
import random as rand
import tensorflow as tf
import numpy as np
from gym.wrappers import AtariPreprocessing, FrameStack

from blurr import to_blurred





def create_trajectory(model, blurrer, batch : int, epsilon : float, epsilon_blurrer : float, env_name : str):
    """
    Create a trajectory for the given model, blurrer and environment. Some hyperparameters are already fixed. Images are always resized to 84x84, grayscaled 
    and stacked in the last dimension in batches of 4.
    Args:
        dqn <DQN> : DQN network for predicting the actions
        blurrer <Blurrer> : Blurrer for predicting the gaze positions
        batch <int> : From how many environments should be sampled. Note that this is not happening in parallel. This is just for increasing the batch size of the model
        epsilon <float> : epsilon value for the epsilon-greedy policy
        env_name <str> : Environment name. Has to exist in the gym package
        frame_skips <int> : How many frames should be skipped
    """

    s_a_r_s = []

    # create action space
    ACTION_SPACE_SIZE = gym.make(env_name,full_action_space=False,new_step_api=True).action_space.n # only discrete

    #  create number of environments equal to batch size
    envs = [FrameStack(AtariPreprocessing(gym.make(env_name,full_action_space=False,new_step_api=True,frameskip=1), frame_skip=4, grayscale_obs = True, new_step_api=True,noop_max=0), num_stack=4, new_step_api=True) for _ in range(batch)]

    # create starting frame for each environment

    observations = [np.transpose(env.reset(),(1,2,0)) for env in envs]

    current_gaze_x = [[42.0,42.0,42.0,42.0] for _ in range(batch)]
    current_gaze_y = [[42.0,42.0,42.0,42.0] for _ in range(batch)]

    for _ in range(10000):

        if not len(observations) == len(envs):
            raise Exception("Length of lists 'observations' and 'envs' did not match")
        if not len(current_gaze_x) == len(envs):
            raise Exception("Length of lists 'current_gaze_x' and 'envs' did not match")
        if not len(current_gaze_y) == len(envs):
            raise Exception("Length of lists 'current_gaze_y' and 'envs' did not match")


        # create list that determines which actions will be sampled and which will be done with the model
        action_selector = [rand.randint(0,99)<epsilon*100 for _ in range(len(envs))]

        # create list that determines which gaze positions will be sampled and which will be done with the blurrers positions
        gaze_selector = [rand.randint(0,99)<epsilon_blurrer*100 for _ in range(len(envs))]

        # to avoid tf.function raytracing, ensure that always a tensor of the same shape is passed to the model
        while len(observations) < batch:
            observations.append(np.zeros(np.shape(observations[0])))
            current_gaze_x.append([0.0,0.0,0.0,0.0])
            current_gaze_y.append([0.0,0.0,0.0,0.0])

        # calculate the actions given the model using a batch and
        # blur all observations and retrieve next gaze position
        with tf.device("/GPU:0"):
            blurred_observation = to_blurred(tf.cast(tf.stack(observations,axis = 0),tf.float32),current_gaze_x,current_gaze_y,4)
            actions = tf.argmax(model(blurred_observation,training = False), axis = -1).numpy()
            new_gaze_x,new_gaze_y = blurrer(blurred_observation,training = False)
            new_gaze_x = new_gaze_x.numpy()
            new_gaze_y = new_gaze_y.numpy()

        observations = observations[:len(envs)]
        actions = actions[:len(envs)]
        new_gaze_x = new_gaze_x[:len(envs)]
        new_gaze_y = new_gaze_y[:len(envs)]

        # sample new actions if the action selector says so
        actions = [rand.randint(0,ACTION_SPACE_SIZE-1) if should_sample_action else action for should_sample_action,action in zip(action_selector,actions)]

        new_gaze_x = [[float(rand.randint(0,84)) for _ in range(4)] if should_make_random_gaze else gaze for should_make_random_gaze,gaze in zip(gaze_selector,new_gaze_x)]
        new_gaze_y = [[float(rand.randint(0,84)) for _ in range(4)] if should_make_random_gaze else gaze for should_make_random_gaze,gaze in zip(gaze_selector,new_gaze_y)]


        result_from_env = []
        for env,action in zip(envs,actions):
            new_observation,reward,done,trunc,_ = env.step(action)
            reward = reward/10
            if trunc == True:
                done = True

            new_observation = np.transpose(new_observation,(1,2,0))
            np.random.seed(42)
            new_observation += np.repeat(np.random.uniform(low=0,high=110,size=(84,84,1)).astype(np.uint8),repeats = 4,axis=-1).astype(np.uint8)
            result_from_env.append((new_observation,reward,done))


        done_and_remove_index = []
        for i, elem in enumerate(zip(observations,actions,result_from_env, current_gaze_x,current_gaze_y,new_gaze_x,new_gaze_y)):
            observation,action,(new_observation, reward, done),x,y,new_x,new_y = elem

            s_a_r_s.append((observation.astype(np.uint8),action,reward,new_observation.astype(np.uint8), done,x,y,new_x,new_y))

            observations[i] = new_observation
            if done:
                done_and_remove_index.append(i)
            
        envs = [env for i,env in enumerate(envs) if not i in done_and_remove_index]
        observations = [observation for i,observation in enumerate(observations) if not i in done_and_remove_index]
        current_gaze_x = [gaze for i,gaze in enumerate(new_gaze_x) if not i in done_and_remove_index]
        current_gaze_y = [gaze for i,gaze in enumerate(new_gaze_y) if not i in done_and_remove_index]

        if len(envs) == 0:
            break
    
    return s_a_r_s


