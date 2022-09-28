import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy import ndimage as ndi

def prepare_data(frames_dir, labels_dir, train_size, batch_size, window_size=4):
    """Helper function for preparing a TF dataset from game frames and
    gaze heatmaps"""
    normalisation_layer = tf.keras.layers.Rescaling(1./255)
    frames = tf.keras.utils.image_dataset_from_directory(
          frames_dir,
          labels=None,
          validation_split=None,
          seed=123,
          image_size=(84, 84),
          shuffle=False,
          batch_size=None,
          color_mode='grayscale',
          crop_to_aspect_ratio=False,
        )

    frames = frames.window(window_size, shift=1, drop_remainder=True)
    frames = frames.flat_map(lambda window: window).batch(window_size)
    frames = frames.map(lambda x: tf.squeeze(x, axis=-1))
    frames = frames.map(lambda x: tf.transpose(x, [1, 2, 0])) # make channels last

    labels = tf.keras.utils.image_dataset_from_directory(
      labels_dir,
      labels=None,
      validation_split=None,
      seed=123,
      image_size=(84, 84),
      shuffle=False,
      batch_size=None,
      color_mode='grayscale',
      crop_to_aspect_ratio=False,
    )

    labels = labels.map(lambda x: (normalisation_layer(x)))
    labels = labels.map(lambda x: tf.squeeze(x, axis=-1))

    ds = tf.data.Dataset.zip((frames, labels))
    ds = ds.shuffle(1000)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size)

    train_ds = train_ds.batch(256).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(256).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

@tf.function
def train_step(model, inputs, targets, loss_fn, optimiser, loss_metric):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimiser.apply_gradients(zip(gradients, model.trainable_variables))
    logs = {}
    loss_metric.update_state(targets, predictions)
    logs["loss"] = loss_metric.result()
    return logs

@tf.function
def test_step(model, inputs, targets, loss_fn, loss_metric):
    predictions = model(inputs, training=False)
    loss = loss = loss_fn(targets, predictions)
    logs = {}
    loss_metric.update_state(targets, predictions)
    logs["val_loss"] = loss_metric.result()
    return logs, predictions

def reset_metrics(metrics):
    """Utility function to reset the state of metrics"""
    for metric in metrics:
        metric.reset_state()

def heatmap_comparison_using_AUC(map1, map2, threshold=2):
    """ binary classifier: Area Under ROC Curve
    Threshold value determines the percentile of salient
    pixels converted to 1 for a binary map.
    """

    # normalise map: all values between 0 and 1
    map1_max = tf.reduce_max(map1)
    map2_max = tf.reduce_max(map2)
    map1_normal = map1 / map1_max
    map2_normal = map2 / map2_max
    # flatten maps
    map1_flat = tf.reshape(map1_normal,[-1]).numpy()
    map2_flat = tf.reshape(map2_normal,[-1]).numpy()
    # create binary maps with only zeros and ones
    map1_rounded = round_with_threshold(map1_flat, 2)
    map2_rounded = round_with_threshold(map2_flat, 2)

    auc_score = roc_auc_score(map1_rounded, map2_rounded)

    return auc_score

def round_with_threshold(array, percent):
    """Calculate the nth percentile and convert all values
    to either 0 or 1 using that value. Common values
    used for saliency maps are 2nd, 5th, 10th, 20th percentiles.
    Cf. Le Meur, O., & Baccino, T. (2013). Methods for comparing
    scanpaths and saliency maps: Strengths and weaknesses.
    Behavior Research Methods, 45 (1), 251–266."""

    value = np.percentile(array, 1)
    return np.where(array >= value, 1, 0)
