import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

np.random.seed(510)
s1=np.random.normal(0,1,200)
s2=np.random.normal(0,1,200)
s3=np.random.normal(0,1,200)
x1=s1
x2=s1+0.001*s2
x3=10*s3

trainData = np.transpose(np.array([x1,x2,x3]))

sess = tf.InteractiveSession()
K = 1
n = 0.01

def Gaussian_prob(X,mu,var):
    Dims = tf.cast(tf.shape(X)[1], tf.float64)
    pi = tf.constant(3.14159265359,tf.float64)
    log_det = tf.reduce_sum(tf.log(tf.square(tf.diag_part(tf.cholesky(var)))))
    first_term = - (Dims/2.)*tf.log(2.*pi) - 0.5 * log_det
    second_term = -0.5 * tf.diag_part(tf.matmul(tf.matmul((X-mu), tf.matrix_inverse(var)), tf.transpose(X-mu)))
    log_gauss1 = first_term + second_term
    log_gauss = tf.reduce_sum(log_gauss1)
    
    return log_gauss, first_term

def graph():
    X = tf.placeholder(tf.float64, [None,3], name='input_x')
    Mu =  tf.Variable(tf.truncated_normal(shape=[1,3]), name='Mu')
    W = tf.Variable(tf.truncated_normal(shape=[K,3]), name='weights')
    Phi_ =  tf.Variable(tf.truncated_normal(shape=[3,]), name='phi')
    
    Mu = tf.cast(Mu, tf.float64)
    W = tf.cast(W, tf.float64)
    Phi_ = tf.cast(tf.exp(Phi_), tf.float64)
    
    #a = tf.cast(tf.zeros([3,3]),tf.float64)
    phi = tf.cast(tf.diag(Phi_), tf.float64)
    #phi = tf.cast(tf.matrix_set_diag(tf.cast(tf.zeros([64,64]),tf.float64), tf.abs(Phi_)), tf.float64)
    var = phi + tf.matmul(tf.transpose(W),W)
    
    
    log_px, f = Gaussian_prob(X, Mu, var)
    #logpxsum = tf.reduce_sum(log_px)
    loss = -log_px
    
    optimizer = tf.train.AdamOptimizer(learning_rate=n,beta1=0.9,beta2=0.99,epsilon=1e-5)
    train = optimizer.minimize(loss)
    
    return train, loss, X, Mu, W, log_px, f
    
    
def runMult():
    train_err = []
    #valid_err = []
    logpxsum_train = []
    #logpxsum_valid = []
    # Build computation graph
    train, loss, data, Mu, W, var, f = graph()

    # Initialize session
    init = tf.initialize_all_variables()
    sess.run(init)

    #train model
    for i in np.arange(5000):
        _, err, mean, weight, variance, first = sess.run([train, loss, Mu, W, var, f], feed_dict={data:trainData})
        train_err.append(err)
        logpxsum_train.append(variance)
        print('update_times:',i)
        #print('update_times:',i,'TRAIN_v:',variance)
        #print('mean:',mean)
        print('[x1 x2 x3] = ',weight)
        #print('var:',mean)
        #print(sess.run(tf.shape(variance)))
        
#        errValid, varvalid = sess.run([loss, var], feed_dict= {data: validData})
#        valid_err.append (errValid)
#        logpxsum_valid.append(varvalid)
        #print('update_times:',i,'VALID_loss:',errValid)
        #print('update_times:',i,'valid_v:',varvalid)
    
    kk = trainData * weight
    ax.scatter(kk[:, 0], kk[:, 2], kk[:, 1])
    ax.set_xlabel("x1", fontsize=20)
    ax.set_ylabel("x3", fontsize=20)
    ax.set_zlabel("x2", fontsize=20)

    plt.ylim([-4, 4])
    #plt.plot(train_err)

    #plt.plot(valid_err)
    plt.show()
    #plt.plot(logpxsum_train)
    #plt.plot(logpxsum_valid)
    
#    assign = np.int32(assign)
#    print('class:', assign)
#    x = trainData[:, 0]
#    y = trainData[:, 1]
#    
#    plt.scatter(x,y,c=assign)
    

runMult()


