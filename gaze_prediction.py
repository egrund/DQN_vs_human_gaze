# Gaze prediction network following Zhang et al. (2019)

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, \
BatchNormalization, Softmax, MaxPooling2D

class GazePrediction(Model):
    def __init__(self, env, frame_stack=4):
        super().__init__()
        input_channels = frame_stack
        self.norm1     = BatchNormalization()
        self.conv1     = Conv2D(filters=32, kernel_size=(8, 8), strides=4,
                                padding="valid", activation='relu',
                                kernel_initializer='he_normal')
        self.conv2     = Conv2D(filters=64, kernel_size=(4, 4), strides=2,
                                padding="valid", activation='relu',
                                kernel_initializer='he_normal')
        self.norm2     = BatchNormalization()
#         self.pool1     = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")
        self.conv3     = Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                                padding="valid", activation='relu',
                                kernel_initializer='he_normal')
#         self.pool2     = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")
        self.norm3     = BatchNormalization()

        self.deconv1   = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=1,
                                padding="valid", activation='relu',
                                kernel_initializer='he_normal')
        self.deconv2   = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=2,
                                padding="valid", activation='relu',
                                kernel_initializer='he_normal')
        self.deconv3   = Conv2DTranspose(filters=1, kernel_size=(8, 8), strides=4,
                                padding="valid", activation='relu',
                                kernel_initializer='he_normal')
        self.softmax   = Softmax(axis=2)

    def call(self, obs, training=True):
        obs = obs / 0xFF
        output = self.conv1(obs, training=training)
        output = self.norm1(output, training=training)
        output = self.conv2(output, training=training)
#         output = self.norm2(output, training=training)
        output = self.conv3(output, training=training)
#         output = self.norm3(output, training=training)

        output = self.deconv1(output, training=training)
        output = self.norm2(output, training=training)
        output = self.deconv2(output, training=training)
        output = self.deconv3(output, training=training)
        output = self.softmax(output)
        return output


if __name__ == '__main__':

    import gym
    from gym.wrappers import FrameStack, AtariPreprocessing
    import tensorflow as tf

    env = gym.make('ALE/Asterix-v5', frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4)
    # frame stacking with 4 frames
    env = FrameStack(env, num_stack=4)
    model = GazePrediction(env)
    model(tf.zeros([1, 84, 84, 4]))
    model.summary()
    obs = tf.cast(env.reset(), tf.float32)
    obs = tf.expand_dims(obs, axis=0) # add batch dim
    obs = tf.transpose(obs, [0, 2, 3, 1]) # batch dim, height, width, channels
    print(model.predict(obs).shape)

    # model.save_weights("gaze_prediction", save_format="tf")
