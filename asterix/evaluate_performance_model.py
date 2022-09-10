from model import AgentModel
import gym
import random as rand
import tensorflow as tf
from sample_trajectory import sample_from_env, preprocess_image

def create_full_trajectory(model,env = gym.make("ALE/Asterix-v5",full_action_space=False,new_step_api=True),frame_skips = 4,imgx = 84, imgy = 84):
    ACTION_SPACE_SIZE = env.action_space.n # only discrete
    performance = 0
    observation = env.reset()
    observation = preprocess_image(observation,imgx ,imgy)
    observation = tf.repeat(observation,frame_skips,axis=-1)
    for _ in range(15000):
        action = tf.argmax(tf.squeeze(model(tf.expand_dims(observation,axis=0),training = False),axis = 0)).numpy()
        new_observation, reward, done = sample_from_env(env, action, imgx, imgy, stack_frames=frame_skips, noops = 0)
        performance += reward
        observation = new_observation
        if done:
            return performance, True
    return performance, False

if __name__ == '__main__':
    model = AgentModel()
    model.load_weights('asterix_test/run8/model')

    its = 100
    performance = 0
    done = []
    for i in range(its):
        p,d = create_full_trajectory(model)
        performance += p
        done.append(d)
        print(i," : Performance= ",p," Done= ",d)
    performance = performance / its

    print("average performance: ", performance)