from sklearn.metrics import roc_auc_score
import tensorflow as tf
import numpy as np

def heatmap_comparison_using_AUC(map1, map2):
    """ binary classifier: Area Under ROC Curve """

    # make map between 0 and 1
    map1_max = tf.reduce_max(map1)
    map2_max = tf.reduce_max(map2)
    map1_normal = map1 / map1_max
    map2_normal = map2 / map2_max
    # make map1 only 0 or 1 
    map1_rounded = round_with_threshold(map1_normal,0.2)
    map2_rounded = round_with_threshold(map2_normal,0.2)
    # flatten both
    map1_flat = tf.reshape(map1_rounded,[-1]).numpy()
    map2_flat = tf.reshape(map2_rounded,[-1]).numpy()

    auc_score = roc_auc_score(map1_flat, map2_flat)

    return auc_score, map1_rounded, map2_rounded

def round_with_threshold(array,threshold=0.5, min=0, max=1):
    return np.where(array > threshold, max, min)
