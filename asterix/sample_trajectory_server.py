import gym
import random as rand
import tensorflow as tf
import numpy as np
import socket
from multiprocessing import Process
import os
import pickle
from buffer import Buffer, BufferManager
from multiprocessing.managers import SyncManager

from model import AgentModel

def preprocess_image(image, imgx,imgy):

    image = tf.image.resize(image, (imgx,imgy))
    return image

def sample_from_env(env : gym.Env, action : int, imgx : int, imgy : int, stack_frames : int, noops : int):
    
    observation, reward, done,_trunc, _ = env.step(action)
    observation = preprocess_image(observation, imgx, imgy)

    for i in range(stack_frames-1):

        
        # if trajectory ended, end normal frame skipping
        if done == True:

            if i < stack_frames-1:

                new_observation = tf.repeat(tf.expand_dims(observation[:,:,-1],axis=-1),(stack_frames-1-i)*3,axis=-1)
                observation = tf.concat([observation,new_observation],axis=-1)
            break

        new_observation, new_reward, done, _trunc, _  = env.step(noops)
        new_observation = preprocess_image(new_observation, imgx, imgy)



        observation = tf.concat([observation,new_observation],axis=-1)
        reward += new_reward

    
    return observation,reward,done


def create_trajectory(model, batch : int, epsilon : float,env_name : str ,frame_skips : int,imgx : int, imgy : int):


    s_a_r_s = []

    # create action space
    ACTION_SPACE_SIZE = gym.make(env_name,full_action_space=False,new_step_api=True).action_space.n # only discrete

    #  create number of environments equal to batch size
    envs = [gym.make(env_name,full_action_space=False,new_step_api=True) for _ in range(batch)]

    # create starting frame for each environment
    observations = [preprocess_image(env.reset(),imgx, imgy) for env in envs]
    observations = [tf.repeat(observation,frame_skips,axis=-1) for observation in observations]



    for _ in range(10000):
        

        if not len(observations) == len(envs):
            raise Exception("Length of lists 'observation' and 'envs' did not match")

        # create list that determines which actions will be sampled and which will be done with the model
        action_selector = [rand.randint(0,100)<=epsilon*100 for _ in range(len(envs))]

        # to avoid tf.function raytracing, ensure that always a tensor of the same shape is passed to the model
        while len(observations) < batch:
            observations.append(tf.zeros(tf.shape(observations[0])))


        # calculate the actions given the model using a batch
        with tf.device("/GPU:0"):

            actions = tf.argmax(model(tf.stack(observations,axis = 0),training = False), axis = -1).numpy().astype(np.int16)


        observations = observations[:len(envs)]
        actions = actions[:len(envs)]

        # sample new actions if the action selector says so
        actions = [rand.randint(0,ACTION_SPACE_SIZE-1) if should_sample_action else action for should_sample_action,action in zip(action_selector,actions)]

        result_from_env = []
        for env,action in zip(envs,actions):
            observation,reward,done = sample_from_env(env, action, imgx , imgy, stack_frames = frame_skips, noops = 0)
            result_from_env.append((observation,reward,done))

        done_and_remove_index = []
        for i, elem in enumerate(zip(observations,actions,result_from_env)):
            observation,action,(new_observation, reward, done) = elem
            s_a_r_s.append((observation.numpy().astype(np.int16),action,reward,new_observation.numpy().astype(np.int16), done))
            observations[i] = new_observation
            if done:
                done_and_remove_index.append(i)
            
        envs = [env for i,env in enumerate(envs) if not i in done_and_remove_index]
        observations = [observation for i,observation in enumerate(observations) if not i in done_and_remove_index]


        if len(envs) == 0:
            break
    
    
    
    return s_a_r_s


if __name__ == "__main__":
    soc = socket.socket()
    soc.bind(('localhost',7994))
    soc.listen(1)
    while True:
        print("waiting...")
        con,_ = soc.accept()
        print("accepted receive")
        with con:
            fragments = []
            while True:
                fragment = con.recv(4096)
                if not fragment: break
                fragments.append(fragment)
            data = pickle.loads(b''.join(fragments))
        
        weights = data["weights"]
        batch = data["batch"]
        epsilon = data["epsilon"]
        env_name = data["env_name"]
        frame_skips = data["frame_skips"]
        imgx = data["imgx"]
        imgy = data["imgy"]

        model = AgentModel(9)
        model(tf.zeros((1,84,84,12)))
        model.set_weights(weights)
        print("sampling")
        new_data = create_trajectory(model, batch, epsilon, env_name, frame_skips, imgx, imgy)
        print("waiting...")
        con,_ = soc.accept()
        print("accepted send")
        with con:
            con.sendall(pickle.dumps(new_data))


