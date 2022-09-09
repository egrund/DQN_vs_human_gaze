from keras import Model
import tensorflow as tf

class AgentModel(Model):

    def __init__(self, num_actions = 9.0, input_shape = (84,84,4)):

        super(AgentModel, self).__init__()

        self._mse = tf.keras.losses.MeanSquaredError()

        #self._resize = tf.keras.layers.Resizing(84,84,interpolation='bilinear',crop_to_aspect_ratio=False,)
        self._l1 = tf.keras.layers.Conv2D(64, (8, 8), strides = (2,2), activation='relu', input_shape=input_shape)#, kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2=0.1))
        self._l2 = tf.keras.layers.MaxPooling2D((3, 3))
        self._l3 = tf.keras.layers.Conv2D(64, (4, 4), activation='relu')#, kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2=0.1))
        self._l4 = tf.keras.layers.MaxPooling2D((3, 3))
        self._l5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')#, kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2=0.1))
        self._l6 = tf.keras.layers.GlobalMaxPooling2D()
        self._l7 = tf.keras.layers.Dense(512,activation="relu")#, kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.1, l2=0.1))

        # state value
        self._l10 = tf.keras.layers.Dense(1,activation="linear")

        # advantages
        self._num_actions = num_actions
        self._l11 = tf.keras.layers.Dense(self._num_actions,activation="linear")

    @tf.function
    def call(self,x ,training):

        x = tf.cast(x, tf.float32)
        x = (x-127.5)/127.5

        x = tf.nn.conv2d(x, filters = [[[[0.299,0.0,0.0,0.0],
        [0.587,0.0,0.0,0.0],
        [0.114,0.0,0.0,0.0],
        [0.0,0.299,0.0,0.0],
        [0.0,0.587,0.0,0.0],
        [0.0,0.114,0.0,0.0],
        [0.0,0.0,0.299,0.0],
        [0.0,0.0,0.587,0.0],
        [0.0,0.0,0.114,0.0],
        [0.0,0.0,0.0,0.299],
        [0.0,0.0,0.0,0.587],
        [0.0,0.0,0.0,0.114]]]],strides = 1,padding = "VALID")
        
        x = self._l1(x,training=training)
        x = self._l2(x,training=training)
        x = self._l3(x,training=training)
        x = self._l4(x,training=training)
        x = self._l5(x,training=training)
        x = self._l6(x,training=training)
        x = self._l7(x,training=training)

        state_value = self._l10(x,training=training)
        advantages = self._l11(x,training=training)

        return state_value + advantages - tf.expand_dims(tf.reduce_sum(advantages,axis = -1)/tf.cast(self._num_actions,tf.float32),axis = -1)
    
    @tf.function
    def step(self,s,a,r,s_new,done,optimizer,model_target):
        

        with tf.GradientTape() as tape:

            with tf.device("/GPU:0"):

                # calculate the corresponding q values
                Q_max = tf.math.reduce_max(model_target(s_new),axis=1)
                Q_s_a = tf.gather(params = self(s),indices = tf.cast(a,tf.int32),axis = 1,batch_dims = 1)

                # apply mean squared error loss
                loss = self._mse(Q_s_a, r + (tf.constant(0.99)*Q_max)*(1-done))

        # perform gradient descent step
        grads = tape.gradient(loss, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss
