# Gaze prediction network following Zhang et al. (2019)
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, \
BatchNormalization, Softmax, Activation

class GazePrediction(Model):
    def __init__(self, env, frame_stack=4):
        super().__init__()
        input_channels = frame_stack
        self.conv1     = Conv2D(filters=32, kernel_size=(8, 8), strides=4,
                                padding='valid', activation=None,
                                kernel_initializer='he_normal')
        self.norm1     = BatchNormalization()
        self.relu1     = Activation(tf.nn.relu)

        self.conv2     = Conv2D(filters=64, kernel_size=(4, 4), strides=2,
                                padding='valid', activation=None,
                                kernel_initializer='he_normal')
        self.norm2     = BatchNormalization()
        self.relu2     = Activation(tf.nn.relu)

        self.conv3     = Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                                padding='valid', activation=None,
                                kernel_initializer='he_normal')
        self.norm3     = BatchNormalization()
        self.relu3     = Activation(tf.nn.relu)

        self.deconv1   = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=1,
                                padding='valid', activation=None,
                                kernel_initializer='he_normal')
        self.norm4     = BatchNormalization()
        self.relu4     = Activation(tf.nn.relu)

        self.deconv2   = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=2,
                                padding='valid', activation=None,
                                kernel_initializer='he_normal')
        self.norm5     = BatchNormalization()
        self.relu5     = Activation(tf.nn.relu)

        self.deconv3   = Conv2DTranspose(filters=1, kernel_size=(8, 8), strides=4,
                                padding='valid', activation=None,
                                kernel_initializer='he_normal')
        self.norm6     = BatchNormalization()
        self.relu6     = Activation(tf.nn.relu)
        self.softmax   = Softmax()

    def call(self, obs, training=True):
        input_shape = tf.shape(obs)
        obs = tf.transpose(obs, [0, 2, 3, 1]) # make channels last: NCHW doesn't work on the M1 chip
        obs = obs / 0xFF
        output = self.conv1(obs, training=training)
        output = self.norm1(output, training=training)
        output = self.relu1(output, training=training)
        output = self.conv2(output, training=training)
#         output = self.norm2(output, training=training)
        output = self.relu2(output, training=training)
        output = self.conv3(output, training=training)
#         output = self.norm3(output, training=training)
        output = self.relu3(output, training=training)

        output = self.deconv1(output, training=training)
        output = self.norm2(output, training=training)
        output = self.relu4(output, training=training)
        output = self.deconv2(output, training=training)
        output = self.relu5(output, training=training)
        output = self.deconv3(output, training=training)
#         output = self.norm6(output, training=training)
        output = self.relu6(output, training=training)
        output = self.softmax(tf.reshape(output, [-1]))

        return tf.reshape(output, (input_shape[0], input_shape[2], input_shape[3]))


if __name__ == '__main__':

    import tensorflow as tf
    import gym
    from gym.wrappers import FrameStack, AtariPreprocessing

    env = gym.make('ALE/Asterix-v5', frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4)
    # frame stacking with 4 frames
    env = FrameStack(env, num_stack=4)
    model = GazePrediction(env)
    obs = tf.cast(env.reset(), tf.float32)
    obs = tf.expand_dims(obs, axis=0) # add batch dim
    model(obs)
    model.summary()
