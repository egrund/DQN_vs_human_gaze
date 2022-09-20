# contains several functions for comparing the gaze data as heatmaps or fixation location map or list of locations with a saliency heatmap

from sklearn.metrics import roc_auc_score
import tensorflow as tf
import numpy as np
from scipy import ndimage as ndi 
from statsmodels.stats.weightstats import ztest 

def heatmap_comparison_using_AUC(gaze, saliency):
    """ binary classifier: Area Under ROC Curve 

    Parameters:
        gaze (heatmap or location map): heatmap -> AUC, location map -> AUC-Judd
        saliency (heatmap): classifier
    """

    gaze_flat = to_binary_flat(gaze)
    saliency_flat = to_binary_flat(saliency)

    auc_score = roc_auc_score(gaze_flat, saliency_flat)

    return auc_score

def heatmap_comparison_percentage_saliency_also_true(gaze,saliency):
    """ calculates the percentage of points being one from the saliency map, that are one on the gaze heatmap (True Positive) """

    gaze_flat = to_binary_flat(gaze)
    sal_flat = to_binary_flat(saliency)

    sal_reduced = np.delete(sal_flat, np.where(gaze_flat == 0))

    percentage_true = np.sum(sal_reduced) / sal_reduced.shape[0]
    return percentage_true

def heatmap_comparison_percentage_same(gaze,saliency):

    gaze_flat = to_binary_flat(gaze)
    sal_flat = to_binary_flat(saliency)

    total = gaze_flat.shape[0] # 33600
    same = np.where(gaze_flat == sal_flat,1,0) # condition, true, false
    return np.sum(same) / total

def to_binary_flat(map):
    """ creates a binary flat map from a heatmap """
    # make map between 0 and 1
    map_max = tf.reduce_max(map)
    map_normal = map / map_max
    # make map only 0 or 1 
    map_rounded = round_with_threshold(map_normal)
    # flatten
    map_flat = tf.reshape(map_rounded,[-1]).numpy()
    return map_flat

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
    information_gain = ( 1/n ) * tf.reduce_sum(tf.multiply(fixation_map,log_saliency - log_prior))
    return information_gain.numpy()
    
def create_center_prior_baseline(map):
    """ creates a center prior baseline in the shape of map """
    base = np.zeros(shape=map.shape)
    center_x = base.shape[0] // 2
    center_y = base.shape[1] //2
    base[center_x,center_y] = 1
    base = ndi.gaussian_filter(base, sigma = 7) # compare my_reader_class.create_gaze_heatmap
    return base

def compare_by_mean(gaze_list : list ,saliency_map):
    """ calculates the distance of the mean value for the gaze list and a saliency map. the map made into a binary map first and only 1 values are used for the mean. 
    it also outputs the distance of the gaze to the middle of the image, to see if that would have been a better classifier (only good prediction if dist to saliency smaller than to middle)
    """

    s_map_max = tf.reduce_max(saliency_map)
    s_map_normal = saliency_map / s_map_max
    s_map_rounded = round_with_threshold(s_map_normal)

    gaze_mean = np.mean(gaze_list)
    saliency_list = []
    for x in range(saliency_map.shape[0]):
        for y in range(saliency_map.shape[1]):
            if s_map_rounded[x][y] == 1:
                saliency_list.append((x,y))
    saliency_mean = np.mean(saliency_list)

    # check if the middle would have been a better guess
    numerator = np.mean(saliency_map.shape)
    dist_middle = np.linalg.norm(gaze_mean - np.array([int(saliency_map.shape[0]/2),int(saliency_map.shape[1]/2)])) / numerator
    dist_sal = np.linalg.norm(gaze_mean - saliency_mean) / numerator

    return dist_sal, dist_middle

def calc_correlation(gaze_map,saliency_map):
    """ compares how similar the two input maps are by using correlation """
    gaze_flat = gaze_map.flatten()
    sal_flat = saliency_map.flatten()
    correlation_matrix = np.corrcoef(gaze_flat,sal_flat)
    return correlation_matrix[0,1]

def heatmap_correlation(gaze_map,saliency_map):
    """ calculates a correlation heatmap for the given input 
    both should contains floats in the range of 0-1 """

    cor_heatmap = tf.divide(tf.multiply(saliency_map,gaze_map),  tf.sqrt(tf.reduce_sum( tf.add(tf.square(saliency_map), tf.square(gaze_map)) ) ) )
    return cor_heatmap

def z_test(x,y,alternative='both-sided'):
    """ source: https://stackoverflow.com/questions/61379874/how-to-perform-two-sample-one-tailed-t-test-in-python """

    _, double_p = ztest(x, y, value=0) # value = differenz between the two under H0
    pval = -1
    if alternative == 'both-sided':
        pval = double_p
    elif alternative == 'greater':
        if np.mean(x) > np.mean(y):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    elif alternative == 'less':
        if np.mean(x) < np.mean(y):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    return pval