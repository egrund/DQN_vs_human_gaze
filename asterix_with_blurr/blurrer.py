from keras import Model
import tensorflow as tf

from blurr import to_blurred

class Blurrer(Model):
    """
    This class implements a Blurrer as described in the paper.
    """

    def __init__(self, input_shape = (84,84,4)):

        super(Blurrer, self).__init__()

        self._mse = tf.keras.losses.MeanSquaredError()

        self._l1 = tf.keras.layers.Conv2D(64, (8, 8), strides = (2,2),activation='relu', input_shape=input_shape)
        self._l2 = tf.keras.layers.MaxPooling2D((3, 3))
        self._l3 = tf.keras.layers.Conv2D(64, (4, 4), activation='relu')
        self._l4 = tf.keras.layers.MaxPooling2D((3, 3))
        self._l5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self._l6 = tf.keras.layers.GlobalMaxPooling2D()
        self._l7 = tf.keras.layers.Dense(512,activation="relu")
        initializer = tf.keras.initializers.Constant(value=0.009125)
        self._l10 = tf.keras.layers.Dense(4,activation="linear", bias_initializer='zeros', kernel_initializer = initializer)
        self._l11 = tf.keras.layers.Dense(4,activation="linear", bias_initializer='zeros', kernel_initializer = initializer)

    @tf.function
    def call(self,x ,training):
        x = tf.cast(x,tf.float32)
        x = (x-127.5)/127.5

        x = self._l1(x,training=training)
        x = self._l2(x,training=training)
        x = self._l3(x,training=training)
        x = self._l4(x,training=training)
        x = self._l5(x,training=training)
        x = self._l6(x,training=training)
        x = self._l7(x,training=training)


        return self._l10(x,training=training)*84.0,self._l11(x,training=training)*84.0 
    
    @tf.function
    def step(self,s,s_new_unblurred,optimizer,dqn):
        """
        Perform a training step according to the paper
        Args:
            - s <tf.Tensor> : state sampled from the environment
            - s_new_unblurred <tf.Tensor> : unblurred new state from pefroming a in s
            - optimizer <tf.keras.optimizers.Optimizer> : optimizer used for performing the training step
            - dqn <DQN> : DQN or DQN_target, depending on the implementation. Used to calculate q-values
        Returns:
            - loss <tf.Tensor> : mse loss
        """
        
        with tf.GradientTape() as tape:

            with tf.device("/GPU:0"):

                x,y = self(s)

                # get blurred state depending on the output of the Blurrer
                s_new_blurred = to_blurred(s_new_unblurred,x,y,4)

                # calculate q values of this new blurred state
                q_values = dqn(s_new_blurred)

            loss = -tf.reduce_mean(tf.square(q_values)*(q_values/tf.abs(q_values))) + tf.reduce_mean(tf.square(-tf.math.minimum(0.0,x))) + tf.reduce_mean(tf.square(-tf.math.minimum(0.0,tf.constant(84.0) - x))) + tf.reduce_mean(tf.square(-tf.math.minimum(0.0,y))) + tf.reduce_mean(tf.square(-tf.math.minimum(0.0,tf.constant(84.0) - y)))

        # perform gradient descent step
        grads = tape.gradient(loss, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss,x,y
