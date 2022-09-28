
from pympler import asizeof
from blurrer import Blurrer
from buffer import Buffer
import agent
from dqn import DQN
from sample_trajectory import create_trajectory
import joblib
import tensorflow as tf 
if __name__ == '__main__':


    # set hyperparameters
    params = {
    "BUFFER_SIZE" : 125000, # max size of buffer
    "BUFFER_MIN" : 125000-1, # min size of buffer that has to be reached before training
    "KEEP_IN_MEM" : True, # if buffer elements should be kept in memory. If false, samples are stored and retrieved from BUFFER_PATH
    "BUFFER_PATH" : "./buffer/", # path to where the buffer data should be stored
    "USE_PRIORITIZED_REPLAY" : False, # should prioritized replay be used
    "EPSILON_DECAY" : 0.002, # how much to decay epsilon each iteration
    "EPSILON_DECAY_BLURRER" : 0.003,
    "START_BLURRER_AT" : 15,
    "INNER_ITS" : 60, # how many training steps per iteration
    "SAMPLES_FROM_ENV" : 3000, # how many new samples from the environment should be added to the buffer each iteration (in expectation)
    "TRAIN_ITS" : 12000, # how many training iterations should be done
    "INITIAL_EPSILON" : 1, # initial value of epsilon
    "EPSILON_MIN" : 0.1, # minimum value of epsilon that can be reached
    "BATCH_SIZE" : 256, # batch size
    "POLYAK_UPDATE" : 0.0102, # polyak update for each iteration
    "LEARNING_RATE" : 0.00025, # learning rate for the adam
    "ENV" : "ALE/Asterix-v5", # environment name
    "LOG_PATH_WEIGHTS" : 'asterix_test/run1/', # where to store the weights
    "LOG_PATH_TENSORBOARD" : 'logs/asterix_test/run1/', # where to store dqn loss and reward for tensorboard
    "PRELOAD_WEIGHTS" : None # path to preloaded weights
    }


 
    try: 
        with open("buffer.pkl" , "rb") as f:
            buffer = joblib.load(f)
    except:

        # fill the buffer
        buffer = Buffer(params["BUFFER_SIZE"], params["BUFFER_MIN"], params["BUFFER_PATH"], params["KEEP_IN_MEM"], params["USE_PRIORITIZED_REPLAY"])
        blurrer = Blurrer()
        model = DQN(9)
        model(tf.random.uniform(shape=(1,84,84,4)))
        blurrer(tf.random.uniform(shape=(1,84,84,4)))

        if params["PRELOAD_WEIGHTS"]:
            model.load_weights(params["PRELOAD_WEIGHTS"] + "model")
            blurrer.load_weights(params["PRELOAD_WEIGHTS"] + "blurrer")

        model(tf.random.uniform(shape=(1,84,84,4)))
        blurrer(tf.random.uniform(shape=(1,84,84,4)))
        total = 0
        while total < params["BUFFER_MIN"]:
            print("total: ",total)
        
            data = create_trajectory(model,blurrer,10, params["INITIAL_EPSILON"], params["INITIAL_EPSILON"], params["ENV"])
            print(len(data))
            total += len(data)
            buffer.extend(data)
            print("size of buffer in gb: ", asizeof.asizeof(buffer)/1000000000)
            print("gpu memory: ", tf.config.experimental.get_memory_info("/GPU:0"))
        
        with open("buffer.pkl" , "wb") as f:
            joblib.dump(buffer,f)


    DQN_agent = agent.agent(
    buffer = buffer,
    use_prioritized_replay = params["USE_PRIORITIZED_REPLAY"],
    env = params["ENV"], 
    epsilon = params["INITIAL_EPSILON"], 
    epsilon_decay = params["EPSILON_DECAY"], 
    epsilon_decay_blurrer = params["EPSILON_DECAY_BLURRER"],
    start_blurrer_at = params["START_BLURRER_AT"],
    epsilon_min = params["EPSILON_MIN"],
    batch_size = params["BATCH_SIZE"],
    inner_its = params["INNER_ITS"],
    samples_from_env = params["SAMPLES_FROM_ENV"],
    polyak_update = params["POLYAK_UPDATE"],
    learning_rate = params["LEARNING_RATE"],
    preload_weights_path = params["PRELOAD_WEIGHTS"])



    DQN_agent.train(its = params["TRAIN_ITS"], path_model_weights = params["LOG_PATH_WEIGHTS"], path_logs = params["LOG_PATH_TENSORBOARD"])
