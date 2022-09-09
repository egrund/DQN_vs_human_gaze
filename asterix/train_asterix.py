import time
import dqn
import tensorflow as tf
from buffer import Buffer
from model import AgentModel
import socket
import pickle

from sample_trajectory import create_trajectory

if __name__ == '__main__':


    # set hyperparameters
    params = {
    "BUFFER_SIZE" : 500000, # max size of buffer
    "BUFFER_MIN" : 500000-1, # min size of buffer that has to be reached before training
    "KEEP_IN_MEM" : False, # if buffer elements should be kept in memory. If false, samples are stored and retrieved from BUFFER_PATH
    "BUFFER_PATH" : "./buffer/", # path to where the buffer data should be stored
    "USE_PRIORITIZED_REPLAY" : False, # should prioritized replay be used
    "EPSILON_DECAY" : 0.004, # how much to decay epsilon each iteration
    "INNER_ITS" : 20, # how many training steps per iteration
    "SAMPLES_FROM_ENV" : 4000, # how many new samples from the environment should be added to the buffer each iteration (in expectation)
    "TRAIN_ITS" : 12000, # how many training iterations should be done
    "INITIAL_EPSILON" : 0.05, # initial value of epsilon
    "EPSILON_MIN" : 0.05, # minimum value of epsilon that can be reached
    "BATCH_SIZE" : 512, # batch size
    "POLYAK_UPDATE" : 0.0008, # polyak update for each training step (so INNER_ITS polyak updates per training iteration)
    "LEARNING_RATE" : 0.000025, # learning rate for the adam
    "ENV" : "ALE/Asterix-v5", # environment name
    "LOG_PATH_WEIGHTS" : 'asterix_test/run18/', # where to store the weights
    "LOG_PATH_TENSORBOARD" : 'logs/asterix_test/run18/', # where to store dqn loss and reward for tensorboard
    "PRELOAD_WEIGHTS" : 'asterix_test/run8/' # path to preloaded weights
    }

    # send hyperparameters to the buffer
    soc = socket.socket()
    connected = False
    while not connected:
        try: 
            soc.connect(('localhost',7998))
            connected = True
        except:
            time.sleep(5)
            print("Could not connect to 'localhost' with port 7998. Will retry in 5 seconds. Make sure to run buffer_server.py ")
    with soc:
        soc.sendall(pickle.dumps(params))
    
    # fill the buffer
    model = AgentModel(9)
    if params["PRELOAD_WEIGHTS"]:
        model.load_weights(params["PRELOAD_WEIGHTS"] + "model")

    
    for i in range(500):
        print(i)
        data = create_trajectory(model, params["INITIAL_EPSILON"], params["ENV"], 4,84,84,1000)
        soc = socket.socket()
        connected = False
        while not connected:
            try: 
                soc.connect(('localhost',7999))
                connected = True
            except:
                time.sleep(5)
                print("Could not connect to 'localhost' with port 7999. Will retry in 5 seconds ")
        with soc:
            soc.sendall(pickle.dumps({"task" : "extend" , "values" : data}))

    DQN_agent = dqn.agent(
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



    DQN_agent.train(its = params["TRAIN_ITS"], path_model_weights = params["LOG_PATH_WEIGHTS"], path_logs = params["LOG_PATH_TENSORBOARD"])
