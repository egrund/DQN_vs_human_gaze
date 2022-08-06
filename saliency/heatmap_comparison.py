from sklearn.metrics import roc_auc_score
import tensorflow as tf
import numpy as np
from scipy import ndimage as ndi 

def heatmap_comparison_using_AUC(map1, map2):
    """ binary classifier: Area Under ROC Curve """

    # make map between 0 and 1
    map1_max = tf.reduce_max(map1)
    map2_max = tf.reduce_max(map2)
    map1_normal = map1 / map1_max
    map2_normal = map2 / map2_max
    # make map1 only 0 or 1 
    map1_rounded = round_with_threshold(map1_normal)
    map2_rounded = round_with_threshold(map2_normal)
    # flatten both
    map1_flat = tf.reshape(map1_rounded,[-1]).numpy()
    map2_flat = tf.reshape(map2_rounded,[-1]).numpy()

    auc_score = roc_auc_score(map1_flat, map2_flat)

    return auc_score, map1_rounded, map2_rounded

def round_with_threshold(array,threshold=0.2, min=0, max=1):
    return np.where(array > threshold, max, min)

def saliency_information_gain(fixation_map, saliency,epsilon=0.1):
    """ compare by using Information Gain 
    
    Arguments: 
        fixation_map (2D array or Tensor): binary map of fixation points (not a heatmap) 
        saliency: saliency heatmap
        epsilon (float): for regularization
    """
    center_prior = create_center_prior_baseline(fixation_map)

    log_saliency = np.log2(saliency + epsilon)
    log_prior = np.log2(center_prior + epsilon)
    n = tf.reshape(fixation_map,[-1]).shape[0]

    # calculation
    information_gain = ( 1/n ) * tf.reduce_sum(tf.multiply(tf.squeeze(fixation_map,axis=-1),log_saliency - log_prior))
    return information_gain.numpy()
    
def create_center_prior_baseline(map):
    """ creates a center prior baseline in the shape of map """
    base = np.zeros(shape=map.shape)
    center_x = base.shape[0] // 2
    center_y = base.shape[1] //2
    base[center_x,center_y] = 1
    base = ndi.gaussian_filter(base, sigma = 10) # compare my_reader_class.create_gaze_heatmap
    return base



