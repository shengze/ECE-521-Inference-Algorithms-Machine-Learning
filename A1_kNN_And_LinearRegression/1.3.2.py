import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(521)
Data = np.linspace(0.1,10.0,num=100)[:,np.newaxis]
Target = np.sin(Data) + 0.1*np.power(Data,2) + 0.5*np.random.randn(100,1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = Data[randIdx[:80]],Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]],Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]],Target[randIdx[90:100]]
#plt.plot(Data[randIdx[:80]],Target[randIdx[:80]],".")


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
    R = np.zeros((Size_h,Size_w),np.float64)
    R[row_indices,indices] = 1/k
    R = tf.convert_to_tensor(R)
    return R
    
def buildGraph(k):
    # Variable creation
    X = tf.constant(trainData)
    X_star = tf.constant(testData)
    y_target = tf.constant(trainTarget)
    y_hat = tf.constant(testTarget)
  
    # Graph definition9  

    r = Choose_neigh(X,X_star,k)
    
    y_predicted =  tf.matmul(r,y_target)
    
    

   # Error definition
    meanSquareError = tf.reduce_mean(tf.square(y_predicted - y_hat))./2
    return meanSquareError,X,y_target,y_predicted

  

#Z=np.linspace(0.0,11.0,num=1000)[:,np.newaxis]
for k in [1,3,5,50]:
    
    meanSquareError,X,y_target,y_predicted = buildGraph(k)
    print("k=",k,"MSE=",sess.run(meanSquareError))

#plt.plot(trainData,trainTarget,'o')
#plt.plot(Z,sess.run(y_predicted),"-")
#
