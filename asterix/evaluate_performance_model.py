import dqn 
import tensorflow as tf
from sample_trajectory import create_trajectory

model = dqn.model()
model(tf.random.uniform(shape=(1,84,84,4)),training = False)
model.load_weights('asterix_test/run2/model')

its = 10
performance = 0
for _ in range(its):
    _, p = create_trajectory(model)
    performance += p
performance = performance / its

print("average performance: ", performance)