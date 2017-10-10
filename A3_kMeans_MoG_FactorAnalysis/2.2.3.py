import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#tf.set_random_seed(521)
input_data = np.load('data2D.npy')
#np.random.seed(521)
randIndx = np.arange(len(input_data))
np.random.shuffle(randIndx)
input_data = input_data[randIndx]
trainData = input_data[:6667]
validData = input_data[6667:]

sess = tf.InteractiveSession()
data_num = np.shape(input_data)[0]
data_dim = np.shape(input_data)[1]
K=5
n=0.1

def Euclidean_dis(X, Y):
    #return the matrix containing the pairwise Euclidean distances
    
    X_b = tf.expand_dims(X, 1)
    result = X_b - Y
    result_square = tf.square(result)
    Euclidean_dis = tf.reduce_sum(result_square, 2)

    return Euclidean_dis
    
def Gaussian_prob(X,mu,Vsigma):
    
    Dims = tf.cast(tf.shape(X)[1],tf.float64)
    pi = tf.constant(3.14159265359,tf.float64)
    dis = Euclidean_dis(X,mu)
    first_term = -Dims/2.*tf.log(2.*pi*Vsigma*Vsigma)
    second_term = tf.div(-dis,2.*tf.square(Vsigma))
    log_gauss = first_term + second_term

    
    return log_gauss



def jointprob(X,mu,prior,sigma):
    log_gauss = Gaussian_prob(X,mu,sigma)
    posterior = log_gauss+prior
    assignment = tf.arg_max(posterior, 1)
    return posterior, assignment

def reduce_logsumexp(input_tensor, reduction_indices=1, keep_dims=False):

  max_input_tensor1 = tf.reduce_max(input_tensor, reduction_indices, keep_dims=keep_dims)
  max_input_tensor2 = max_input_tensor1
  if not keep_dims:
    max_input_tensor2 = tf.expand_dims(max_input_tensor2, 
                                       reduction_indices) 
  return tf.log(tf.reduce_sum(tf.exp(input_tensor - max_input_tensor2), 
                                reduction_indices, keep_dims=keep_dims)) + max_input_tensor1

def logsoftmax(input_tensor):
  return input_tensor - reduce_logsumexp(input_tensor, keep_dims=True)

def log_prob(x,mean,prior,sigma):
   log_post = jointprob(x,mean,prior,sigma)
   posteriori = logsoftmax(log_post) 
   return posteriori
  
def graph():
    data = tf.placeholder(tf.float64, [None,data_dim], name='input_x')   
    mean =  tf.cast(tf.Variable(tf.random_normal([K, data_dim])),tf.float64)
    sigma = tf.cast(tf.Variable(tf.random_normal([K,])),tf.float64)
    sigma_exp = tf.exp(sigma)
    prior = tf.cast(tf.Variable(tf.random_normal([1,K])),tf.float64)
    prior_softmax = logsoftmax(prior)
    
    #define the graph
    log_post, assignment = jointprob(data,mean,prior_softmax,sigma_exp)
    marginprob = reduce_logsumexp(log_post,keep_dims = True)
    loss_f =-1* tf.reduce_sum (marginprob)

    #define train algorithm
    optimizer = tf.train.AdamOptimizer(learning_rate=n,beta1=0.9,beta2=0.99,epsilon=1e-5)
    train = optimizer.minimize(loss=loss_f)

    return train,data,mean,loss_f, assignment
    
def runMult():
    train_err = []
    valid_err = []
    # Build computation graph
    train,data,mean,loss, assignment = graph()

    # Initialize session
    init = tf.initialize_all_variables()
    sess.run(init)

    #train model
    for i in np.arange(2000):
        _, err, center, assign= sess.run([train, loss, mean, assignment], feed_dict={data:trainData})
        train_err.append(err)
        print('update_times:',i,'TRAIN_loss:',err)
        #print('mean:',center)
        
        errValid = sess.run(loss, feed_dict= {data: validData})
        valid_err.append (errValid)
        print('update_times:',i,'VALID_loss:',errValid)
    
    #plt.plot(train_err)
    #plt.plot(valid_err)
    
    assign = np.int32(assign)
    print('class:', assign)
    x = trainData[:, 0]
    y = trainData[:, 1]
    
    plt.scatter(x,y,c=assign)
    plt.show()
    

runMult()

