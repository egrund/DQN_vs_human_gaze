import dqn

BUFFER_SIZE = 100000
EPSILON_DECAY = 0.09
BUFFER_MIN = 38000
THREADS = 10
INNER_ITS = 80
TRAIN_ITS = 12000

DQN_agent = dqn.agent(epsilon_decay= EPSILON_DECAY,inner_its = INNER_ITS,buffer_size=BUFFER_SIZE,buffer_min=BUFFER_MIN,threads=THREADS)

DQN_agent.train(its=TRAIN_ITS, path_model_weights = 'asterix_test/run1', path_logs = 'logs/asterix_test/run1')

average_performance = DQN_agent.evaluate(100)

print("Average Performance: ", average_performance)