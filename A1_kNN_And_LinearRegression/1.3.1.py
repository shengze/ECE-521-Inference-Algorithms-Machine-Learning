import tensorflow as tf
import numpy as np

sess=tf.InteractiveSession()
def Euclidean_dis(X,Y):
    '''
    return the matrix containing the pairwise Euclidean distances
    '''
    
    X_b = tf.expand_dims(X,1)
    result = X_b - Y
    result_square = tf.square(result)
    Euclidean_dis = tf.reduce_sum(result_square,2)
    
    return Euclidean_dis

    
def Choose_neigh(X,x_target,k):
    
    Dis = Euclidean_dis(X,x_target)
    Dis_t = tf.transpose(Dis)
    Dis_t = Dis_t * (-1)
    values,indices = tf.nn.top_k(Dis_t, k)
    Size_w = tf.shape(X)[0].eval()
    Size_h = tf.shape(x_target)[0].eval()
    indices = sess.run(indices)
    row_indices = np.linspace(0,Size_h-1,Size_h,dtype = int)
    row_indices = row_indices.repeat(k)
    indices = indices.reshape(Size_h*k,)
    R = np.zeros((Size_h,Size_w),np.float32)
    R[row_indices,indices] = 1/k
    R = tf.convert_to_tensor(R)
    return R

