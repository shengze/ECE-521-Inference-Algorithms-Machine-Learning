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

sess = tf.InteractiveSession()

def Euclidean_dis(X,Y):
    '''
    return the matrix containing the pairwise Euclidean distances
    '''
     
    X_b = tf.expand_dims(X,1)
    result = X_b - Y
    result_square = tf.square(result)
    Euclidean_dis = tf.reduce_sum(result_square,2)
    return Euclidean_dis

def squared_exponential(X,x_star,lamda):
    K = tf.exp(Euclidean_dis(X, x_star) * (-1) * lamda)
    return K
    
def soft_kNN(K):
    r_kNN = tf.transpose(K / tf.reduce_sum(K,0))
    return r_kNN
    
def Gaussian_process(K_inv,K):
    K_inv = tf.matrix_inverse(K_inv)
    r_G = tf.transpose(tf.matmul(K_inv,K))
    return r_G
    
X = tf.constant(trainData)
x_star = tf.constant(testData)
y_target = tf.constant(trainTarget)
y_hat = tf.constant(testTarget)

lamda = 100
    # Graph definition9
K = squared_exponential(X,x_star,lamda)

K_inv = squared_exponential(X,X,lamda)
    
r_kNN = soft_kNN(K)
r_G = Gaussian_process(K_inv, K)
    
y_predicted_kNN =  tf.matmul(r_kNN,y_target).eval()
y_predicted_G = tf.matmul(r_G,y_target).eval()

np.reshape(y_predicted_kNN,(10))
np.reshape(y_predicted_G,(10))
print(y_predicted_G)
print(y_predicted_kNN)
#plt.plot(trainData,trainTarget,".")
#plt.plot(testData,y_predicted_kNN,"*")
#plt.plot(testData,y_predicted_G,"*")

    
