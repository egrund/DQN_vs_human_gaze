
import agent
from sample_trajectory import create_trajectory
import joblib
from pympler import asizeof
from buffer import Buffer
from dqn import DQN
import tensorflow as tf
if __name__ == '__main__':


    # set hyperparameters
    params = {
    "BUFFER_SIZE" : 200000, # max size of buffer
    "BUFFER_MIN" : 200000-1, # min size of buffer that has to be reached before training
    "KEEP_IN_MEM" : True, # if buffer elements should be kept in memory. If false, samples are stored and retrieved from BUFFER_PATH
    "BUFFER_PATH" : "./buffer/", # path to where the buffer data should be stored
    "USE_PRIORITIZED_REPLAY" : True, # should prioritized replay be used
    "EPSILON_DECAY" : 0.003, # how much to decay epsilon each iteration
    "INNER_ITS" : 50, # how many training steps per iteration
    "SAMPLES_FROM_ENV" : 3000, # how many new samples from the environment should be added to the buffer each iteration (in expectation)
    "TRAIN_ITS" : 12000, # how many training iterations should be done
    "INITIAL_EPSILON" : 1, # initial value of epsilon
    "EPSILON_MIN" : 0.1, # minimum value of epsilon that can be reached
    "BATCH_SIZE" : 512, # batch size
    "POLYAK_UPDATE" : 0.0102, # polyak update for each iteration
    "LEARNING_RATE" : 0.00025, # learning rate for the adam
    "ENV" : "ALE/Asterix-v5", # environment name
    "LOG_PATH_WEIGHTS" : 'asterix_test/run1/', # where to store the weights
    "LOG_PATH_TENSORBOARD" : 'logs/asterix_test/run1/', # where to store dqn loss and reward for tensorboard
    "PRELOAD_WEIGHTS" : None # path to preloaded weights
    }

    try:
        # try to reload previously saved buffer
        with open("buffer.pkl" , "rb") as f:
            buffer = joblib.load(f)
    except:
        # if this fails make new buffer
        print("Could not load buffer")

        # initialize buffer and model
        buffer = Buffer(params["BUFFER_SIZE"], params["BUFFER_MIN"], params["BUFFER_PATH"], params["KEEP_IN_MEM"], params["USE_PRIORITIZED_REPLAY"])

        model = DQN(9)
        # initialize weights
        model(tf.random.uniform(shape=(1,84,84,4)))

        if params["PRELOAD_WEIGHTS"]:
            model.load_weights(params["PRELOAD_WEIGHTS"] + "model")

        current_buffer_size = 0
        while current_buffer_size < params["BUFFER_MIN"]:
        
            data = create_trajectory(model,10, params["INITIAL_EPSILON"], params["ENV"], 4)
            current_buffer_size += len(data)
            buffer.extend(data)
            print("Current buffer size: ",current_buffer_size)
            print("size of buffer in gb: ", asizeof.asizeof(buffer)/1000000000)
        
        # save buffer when done with filling
        with open("buffer.pkl" , "wb") as f:
            joblib.dump(buffer,f)


    # initialize a new agent with the hyperparamters
    DQN_agent = agent.agent(
    buffer = buffer,
    use_prioritized_replay = params["USE_PRIORITIZED_REPLAY"],
    env = params["ENV"], 
    epsilon = params["INITIAL_EPSILON"], 
    epsilon_decay = params["EPSILON_DECAY"], 
    epsilon_min = params["EPSILON_MIN"],
    batch_size = params["BATCH_SIZE"],
    inner_its = params["INNER_ITS"],
    samples_from_env = params["SAMPLES_FROM_ENV"],
    polyak_update = params["POLYAK_UPDATE"],
    learning_rate = params["LEARNING_RATE"],
    preload_weights_path = params["PRELOAD_WEIGHTS"])

    # train the dqn
    DQN_agent.train(its = params["TRAIN_ITS"], path_model_weights = params["LOG_PATH_WEIGHTS"], path_logs = params["LOG_PATH_TENSORBOARD"])
