import dqn

DQN_agent = dqn.agent(epsilon_decay= 0.09,inner_its = 80,buffer_size=100000,buffer_min=38000,threads=10)

DQN_agent.train(its=12000, path_model_weights = 'asterix_test/run1', path_logs = 'logs/asterix_test/run1')

average_performance = DQN_agent.evaluate(100)

print("Average Performance: ", average_performance)