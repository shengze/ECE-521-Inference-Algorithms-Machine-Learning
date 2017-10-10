import numpy as np
import tensorflow as tf
def Euclidean_dis(X,Y):
    '''
    return the matrix containing the pairwise Euclidean distances
    '''
    
    X_b = tf.expand_dims(X,1)
    result = X_b - Y
    result_square = tf.square(result)
    Euclidean_dis = tf.reduce_sum(result_square,2)
    return Euclidean_dis
 