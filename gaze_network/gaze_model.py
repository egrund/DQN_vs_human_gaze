from keras import Model
import tensorflow as tf
from gaze_prediction import GazePrediction

class AgentModel(Model):
    def __init__(self, num_actions = 9.0, input_shape = (84, 84, 4)):
        super().__init__()

        self._mse = tf.keras.losses.MeanSquaredError()

        self.gaze_network = GazePrediction()
        self.gaze_network.load_weights("gaze_predict")

        ###  4-frame stack channel

        self._l1 = tf.keras.layers.Conv2D(64, (8, 8), strides = (2,2), activation='relu', input_shape=input_shape)
        self._l2 = tf.keras.layers.MaxPooling2D((3, 3))
        self._l3 = tf.keras.layers.Conv2D(64, (4, 4), activation='relu')
        self._l4 = tf.keras.layers.MaxPooling2D((3, 3))
        self._l5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self._l6 = tf.keras.layers.GlobalMaxPooling2D()
        self._l7 = tf.keras.layers.Dense(512,activation="relu")

        # state value
        self._l10 = tf.keras.layers.Dense(1, activation="linear")

        # advantages
        self._num_actions = num_actions
        self._l11 = tf.keras.layers.Dense(self._num_actions, activation="linear")

        ### gaze channel

        self._l01 = tf.keras.layers.Conv2D(64, (8, 8),
                                          strides = (2,2),
                                          activation='relu')
        self._l02 = tf.keras.layers.MaxPooling2D((3, 3))
        self._l03 = tf.keras.layers.Conv2D(64, (4, 4), activation='relu')
        self._l04 = tf.keras.layers.MaxPooling2D((3, 3))
        self._l05 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self._l06 = tf.keras.layers.GlobalMaxPooling2D()

        self._l07 = tf.keras.layers.Dense(512,activation="relu")

        # state value
        self._l010 = tf.keras.layers.Dense(1, activation="linear")

        # advantages

        self._l011 = tf.keras.layers.Dense(self._num_actions, activation="linear")

        self.average = tf.keras.layers.Average()


    @tf.function
    def call(self, inputs, training):
        inputs = tf.cast(inputs, tf.float32)
        inputs = (inputs-127.5)/127.5

        gaze = self.gaze_network(inputs)

        x = self._l1(inputs,training=training)
        x = self._l2(x,training=training)
        x = self._l3(x,training=training)
        x = self._l4(x,training=training)
        x = self._l5(x,training=training)
        ch1_output = self._l6(x,training=training)

        # pass a masked frame through the network

        masked_x = tf.math.multiply(inputs[:,:,:,-1], gaze)
        masked_x = tf.expand_dims(masked_x, axis=-1)
        ch2_output = self._l01(masked_x,training=training)
        ch2_output = self._l02(ch2_output,training=training)
        ch2_output = self._l03(ch2_output,training=training)
        ch2_output = self._l04(ch2_output,training=training)
        ch2_output = self._l05(ch2_output,training=training)
        ch2_output = self._l06(ch2_output,training=training)

        # take the average
        out = self.average([ch1_output, ch2_output])
        out = self._l7(out, training=training)

        # state value
        state_value = self._l10(out,training=training)
        advantages = self._l11(out,training=training)

        out = state_value + advantages - tf.expand_dims(
                    tf.reduce_sum(advantages,axis = -1)/tf.cast(self._num_actions,tf.float32),
                    axis = -1)

        return out


    @tf.function
    def step(self, s, a, r, s_new, done, optimizer, model_target):


        with tf.GradientTape() as tape:

            with tf.device("/GPU:0"):

                # calculate the corresponding q values
                Q_max = tf.math.reduce_max(model_target(s_new), axis=1)
                Q_s_a = tf.gather(params = self(s), indices = tf.cast(a, tf.int32), axis = 1, batch_dims = 1)

                # apply mean squared error loss
                loss = self._mse(Q_s_a, r + (tf.constant(0.99)*Q_max)*(1-done))

        # perform gradient descent step
        grads = tape.gradient(loss, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss
