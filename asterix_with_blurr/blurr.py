import tensorflow as tf
import tensorflow_addons as tfa


@tf.function
def create_weight_map(center_x,center_y,radius,img_x,img_y):
    """
    This functions creates a tensor with the same shape as the original image, 
    where each entry in the tensor is the distance to the outer lane of a circle around a given point.
    
    Args:
        - center_x <tf.float32>: x coordinate of the circle center
        - center_y <tf.float32>: y coordinate of the circle center
        - radius <tf.float32>: radius of the circle
        - img_x <tf.int32>: x dimension of the image
        - img_y <tf.int32>: y dimension of the image
    """

    # create a tensor with the original size of the image
    # each entry in this tensor is a 2D tuple containing the x and y coordinate of its current location
    img_arr_x = tf.range(0,img_y, dtype=tf.float32)
    img_arr_y = tf.range(0,img_x, dtype=tf.float32)
    a,b = tf.meshgrid(img_arr_y,img_arr_x)

    a = tf.square(tf.repeat(tf.expand_dims(tf.repeat(tf.expand_dims(center_x,axis = -1),repeats=img_y,axis=1),axis = -1),axis = -1,repeats=img_x) - a)
    b = tf.square(tf.repeat(tf.expand_dims(tf.repeat(tf.expand_dims(center_y,axis = -1),repeats=img_y,axis=1),axis = -1),axis = -1,repeats=img_x) - b)
    
    img = tf.math.abs(tf.math.maximum(0.0,tf.sqrt(a+b) - tf.repeat(tf.expand_dims(tf.repeat(tf.expand_dims(radius,axis = -1),repeats=img_y,axis=1),axis = -1),axis = -1,repeats=img_x)))

    img = tf.expand_dims(img,axis = -1)
    img = img

    return img

@tf.function
def to_shape(x,shape):
    x = tf.repeat(tf.expand_dims(x,axis=-1),repeats=shape[1],axis=-1)
    x = tf.repeat(tf.expand_dims(x,axis=-1),repeats=shape[2],axis=-1)
    x = tf.repeat(tf.expand_dims(x,axis=-1),repeats=shape[3],axis=-1)
    return x


tf.function
def to_blurred(img, focus_x, focus_y, channels = 4, gauss_kernel = (15,15), radius = 12.0):
    """
    Applies a radial blur to the image around the focus points.
    This function basically creates a weighted average between the same image but blurred in different strengths.
    The weight for the average is obtained by the inverted weight map.
    So the unblurred image gets a weight map with 1's around the center and smaller values around it and 0's far away from the center.
    The image with maximum blur gets a weight map with 0's around the center and bigger values around it and 1's far away from the center.

    Args:
        - img <tf.Tensor>: image to apply the blur to
        - focus_x <tf.float32>: x coordinate of focus point
        - focus_y <tf.float32>: y coordinate of focus point
    """


    # check if image color channel has more than one dimension
    if channels > 1:
        img = tf.concat(tf.split(img,num_or_size_splits=channels,axis=-1),axis=0)
        focus_x = tf.squeeze(tf.concat(tf.split(focus_x,num_or_size_splits=channels,axis=-1),axis=0),axis=-1)
        focus_y = tf.squeeze(tf.concat(tf.split(focus_y,num_or_size_splits=channels,axis=-1),axis=0),axis=-1)

    # get the batch size
    batch_size = tf.shape(img)[0]

    # get the dimensions of the image
    img_x = tf.shape(img)[1]
    img_y = tf.shape(img)[2]

    # get a blurred variant of the image by applying a gaussian filter
    img2 = tfa.image.gaussian_filter2d(img,filter_shape=gauss_kernel,sigma = 1000)

    map1 = create_weight_map(focus_x,focus_y,tf.repeat(radius,repeats=batch_size),img_x,img_y)

    map1 = (map1 - to_shape(tf.math.reduce_min(map1,axis=(1,2,3)),tf.shape(map1)))/(to_shape(tf.math.reduce_max(map1,axis=(1,2,3)),tf.shape(map1)) - to_shape(tf.math.reduce_min(map1,axis=(1,2,3)),tf.shape(map1)))
    
    # save the original map
    map2 = map1

    # invert the weight map
    map1 = (1 - map1)

    map2 = 50*map2
    
    # make a weighted average from the original image and the blurred image with the two weight maps
    images = [tf.math.multiply(img,map1),tf.math.multiply(img2,map2)]
    img = tf.reduce_sum(images,axis = 0)/(map1+map2)

    if channels > 1:
        img = tf.concat(tf.split(img,num_or_size_splits=channels,axis=0),axis=-1)

    return img