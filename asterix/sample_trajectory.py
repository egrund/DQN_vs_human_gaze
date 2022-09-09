import gym
import random as rand
import time 
import joblib
import numpy as np

def preprocess_image(image, imgx,imgy):
    import tensorflow as tf

    with tf.device("/CPU:0"):

        image = tf.image.resize(image, (imgx,imgy))

    
    return image

def sample_from_env(env : gym.Env, action : int, imgx : int, imgy : int, stack_frames : int, noops : int):
    import tensorflow as tf
    
    observation, reward, done,_trunc, _ = env.step(action)
    observation = preprocess_image(observation, imgx, imgy)

    for i in range(stack_frames-1):

        
        # if trajectory ended, end normal frame skipping
        if done == True:

            if i < stack_frames-1:
                with tf.device("/CPU:0"):
                    new_observation = tf.repeat(tf.expand_dims(observation[:,:,-1],axis=-1),(stack_frames-1-i)*3,axis=-1)
                    observation = tf.concat([observation,new_observation],axis=-1)
            break

        new_observation, new_reward, done, _trunc, _  = env.step(noops)
        new_observation = preprocess_image(new_observation, imgx, imgy)


        with tf.device("/CPU:0"):
            observation = tf.concat([observation,new_observation],axis=-1)
        reward += new_reward

    
    return observation,reward,done



def create_trajectory(model, epsilon : float,env_name : str ,frame_skips : int,imgx : int, imgy : int, its : int):
    import tensorflow as tf

    start_of_function = time.time()

    s_a_r_s = []
    env = gym.make(env_name,full_action_space=False,new_step_api=True)
    #env = AtariPreprocessing(env = env)

    # create action space
    ACTION_SPACE_SIZE = env.action_space.n # only discrete

    observation = env.reset()
    observation = preprocess_image(observation, imgx, imgy)
    # bring start state into format of all other states
    with tf.device("/CPU:0"):
        observation = tf.repeat(observation,frame_skips,axis=-1)

    for k in range(10000):
        
        loop_beginning = time.time()

        # epsilon greedy policy
        if rand.randint(0,100)<=epsilon*100:
            action = rand.randint(0,ACTION_SPACE_SIZE-1)
        else:
            with tf.device("/GPU:0"):
                action = tf.argmax(tf.squeeze(model(tf.expand_dims(observation,axis=0),training = False),axis = 0)).numpy().astype(np.int16)
        
        # sample from environment
        new_observation, reward, done = sample_from_env(env, action,imgx , imgy, stack_frames=frame_skips, noops = 0)

        s_a_r_s.append((observation.numpy().astype(np.int16),action,reward,new_observation.numpy().astype(np.int16), done))
        observation = new_observation
            

        if done:
            if k < its:
                observation = env.reset()
                observation = preprocess_image(observation, imgx, imgy)
                # bring start state into format of all other states
                with tf.device("/CPU:0"):
                    observation = tf.repeat(observation,frame_skips,axis=-1)
            else:
                break
            
    
    return s_a_r_s




