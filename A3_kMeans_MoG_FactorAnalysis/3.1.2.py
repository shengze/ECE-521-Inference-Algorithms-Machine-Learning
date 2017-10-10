import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

with np.load("tinymnist.npz") as data:
    np.random.seed(521)
    trainData, trainTarget = data["x"], data["y"]
    validData, validTarget = data["x_valid"], data["y_valid"]
    testData, testTarget = data["x_test"], data["y_test"]

sess = tf.InteractiveSession()
K = 4
n = 0.1
'''
def Euclidean_dis(X, Y):
    X_b = tf.expand_dims(X, 1)
    result = X_b - Y
    result_square = tf.square(result)
    Euclidean_dis = tf.reduce_sum(result_square, 2)

    return Euclidean_dis
'''
def Gaussian_prob(X,mu,var):
    Dims = tf.cast(tf.shape(X)[1], tf.float64)
    #N = tf.cast(tf.shape(X)[0], tf.float64)
    pi = tf.constant(3.14159265359,tf.float64)
    #dis = Euclidean_dis(X,mu)
    log_det = tf.reduce_sum(tf.log(tf.square(tf.diag_part(tf.cholesky(var)))))
    first_term = - (Dims/2.)*tf.log(2.*pi) - 0.5 * log_det
    second_term = -0.5 * tf.diag_part(tf.matmul(tf.matmul((X-mu), tf.matrix_inverse(var)), tf.transpose(X-mu)))
    log_gauss1 = first_term + second_term
    log_gauss = tf.reduce_sum(log_gauss1)
    
#    for i in range(0,700):
#        second_term = - 0.5 * tf.reshape(tf.matmul(tf.matmul((X[i,:]-mu), tf.matrix_inverse(var)), tf.transpose(X[i,:]-mu)),[1])
#        #kk = tf.reduce_sum(second_term)
#        log_gauss += (first_term + second_term)
    #assignment = tf.arg_max(log_gauss, 1)
    
    return log_gauss, first_term

def graph():
    X = tf.placeholder(tf.float64, [None,64], name='input_x')
    Mu =  tf.Variable(tf.truncated_normal(shape=[1,64]), name='Mu')
    W = tf.Variable(tf.truncated_normal(shape=[K,64]), name='weights')
    Phi_ =  tf.Variable(tf.truncated_normal(shape=[64,]), name='phi')
    
    Mu = tf.cast(Mu, tf.float64)
    W = tf.cast(W, tf.float64)
    Phi_ = tf.cast(tf.exp(Phi_), tf.float64)
    
    #a = tf.cast(tf.zeros([64,64]),tf.float64)
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
    valid_err = []
    logpxsum_train = []
    logpxsum_valid = []
    # Build computation graph
    train, loss, data, Mu, W, var, f = graph()

    # Initialize session
    init = tf.initialize_all_variables()
    sess.run(init)

    #train model
    for i in np.arange(200):
        _, err, mean, weight, variance, first = sess.run([train, loss, Mu, W, var, f], feed_dict={data:trainData})
        train_err.append(err)
        logpxsum_train.append(variance)
        print('update_times:',i,'TRAIN_loss:',err)
        #print('update_times:',i,'TRAIN_v:',variance)
        #print('mean:',mean)
        print('second:',first)
        print('var:',variance)


    print('train_err:',-err)
    valid_err = sess.run(loss,feed_dict={data:validData})
    print('valid_err:',-valid_err)
    test_err = sess.run(loss, feed_dict={data: testData})
    print('test_err:', -test_err)
        #print(sess.run(tf.shape(variance)))
        
#        errValid, varvalid = sess.run([loss, var], feed_dict= {data: validData})
#        valid_err.append (errValid)
#        logpxsum_valid.append(varvalid)
        #print('update_times:',i,'VALID_loss:',errValid)
        #print('update_times:',i,'valid_v:',varvalid)
    
    #plt.plot(train_err)
    #plt.plot(valid_err)
    
    #plt.plot(logpxsum_train)
    #plt.plot(logpxsum_valid)
    
#    assign = np.int32(assign)
#    print('class:', assign)
#    x = trainData[:, 0]
#    y = trainData[:, 1]
#    
#    plt.scatter(x,y,c=assign)
    weight=np.reshape(weight,[4,8,8])
    for ax in xrange(1, 5):
        fig = plt.subplot(2, 2, ax)
        plt.imshow(weight[ax-1], cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()

runMult()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
