
import dqn
import tensorflow as tf
from buffer import Buffer


if __name__ == '__main__':


    # set hyperparameters
    BUFFER_SIZE = 250000 # max size of buffer
    BUFFER_MIN = 250000-1 # min size of buffer that has to be reached before training
    USE_PRIORITIZED_REPLAY = False # should prioritized replay be used
    EPSILON_DECAY = 0.004 # how much to decay epsilon each iteration
    INNER_ITS = 40 # how many training steps per iteration
    SAMPLES_FROM_ENV = 2000 # minimum of how many new samples from the environment should be added to the buffer each iteration
    TRAIN_ITS = 12000 # how many training iterations should be done
    INITIAL_EPSILON = 1 # initial value of epsilon
    EPSILON_MIN = 0.1 # minimum value of epsilon that can be reached
    BATCH_SIZE = 512 # batch size
    POLYAK_UPDATE = 0.000125 # polyak update for each training step (so INNER_ITS polyak updates per training iteration)
    OPTIMIZER = tf.keras.optimizers.Adam(0.0025, beta_1=0.9, beta_2=0.999, epsilon=1e-07) # optimizer that is used
    ENV = "ALE/Asterix-v5" # environment name
    LOG_PATH_WEIGHTS = 'asterix_test/run30' # where to store the weights
    LOG_PATH_TENSORBOARD = 'logs/asterix_test/run30/' # where to store dqn loss and reward for tensorboard

    # initialize buffer
    with tf.device("/CPU:0"):
        buffer = Buffer(BUFFER_SIZE, BUFFER_MIN)
        buffer.load("./buffer/")
        buffer.fill(dqn.model(9),1,ENV)
        #buffer.save("./buffer/")

    DQN_agent = dqn.agent(buffer = buffer, 
    use_prioritized_replay = USE_PRIORITIZED_REPLAY,
    env = ENV, 
    epsilon = INITIAL_EPSILON, 
    epsilon_decay = EPSILON_DECAY, 
    epsilon_min = EPSILON_MIN,
    batch_size = BATCH_SIZE,
    inner_its = INNER_ITS,
    samples_from_env = SAMPLES_FROM_ENV,
    polyak_update = POLYAK_UPDATE,
    optimizer = OPTIMIZER)

    DQN_agent.train(its = TRAIN_ITS, path_model_weights = LOG_PATH_WEIGHTS, path_logs = LOG_PATH_TENSORBOARD)

    average_performance = DQN_agent.evaluate(100)

    print("Average Performance: ", average_performance)
