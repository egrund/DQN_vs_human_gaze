import tensorflow as tf
import gym
import random as rand
import matplotlib as plt
import threading, queue

@tf.function
def preprocess_image(image, imgx, imgy):
    image = tf.cast(image,tf.float32)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image,size=(imgx,imgy))
    return image


def sample_from_env(env, action, imgx, imgy, stack_frames=4, noops=0):
    observation, reward, done, _ = env.step(action)
    observation = preprocess_image(observation,imgx, imgy)

    for i in range(stack_frames-1):
        
        # if trajectory ended, end normal frame skipping
        if done == True:
            if i < stack_frames-1:
                new_observation = tf.repeat(tf.expand_dims(observation[:,:,-1],axis=-1),stack_frames-1-i,axis=-1)
                observation = tf.concat([observation,new_observation],axis=-1)
            break

        new_observation, new_reward, done, _  = env.step(noops)
        new_observation = preprocess_image(new_observation,imgx, imgy)
        observation = tf.concat([observation,new_observation],axis=-1)
        reward += new_reward
    
    return observation,reward,done

def create_trajectory(model,render = False,epsilon = 0.7):
    """
    create trajectory given the model
    """

    FRAME_SKIPS = 4
    IMGX = 84
    IMGY = 84
    ACTION_SPACE_SIZE = 9

    env = gym.make("ALE/Asterix-v5",full_action_space=False)

    s_a_r_s = []

    observation = env.reset()
    observation = preprocess_image(observation,IMGX,IMGY)
    # bring start state into format of all other states
    observation = tf.repeat(observation,FRAME_SKIPS,axis=-1)


    for _ in range(10000):

        # epsilon greedy policy
        if rand.randint(0,100)<=epsilon*100:
            action = rand.randint(0,ACTION_SPACE_SIZE-1)
        else:
            action = tf.argmax(tf.squeeze(model(tf.expand_dims(observation,axis=0),training = False),axis = 0)).numpy()

        # sample from environment
        new_observation, reward, done = sample_from_env(env, action,IMGX, IMGY, stack_frames=FRAME_SKIPS)


        s_a_r_s.append((tf.convert_to_tensor(observation),tf.convert_to_tensor(action),tf.convert_to_tensor(reward),tf.convert_to_tensor(new_observation), done))
        observation = new_observation

        if render:
            plt.imshow(observation)
            plt.show()

        if done:
            return s_a_r_s


    return s_a_r_s

def create_trajectory_thread(model,render,epsilon,q):
    s_a_r_s = create_trajectory(model,render,epsilon)
    q.put(s_a_r_s)
 
