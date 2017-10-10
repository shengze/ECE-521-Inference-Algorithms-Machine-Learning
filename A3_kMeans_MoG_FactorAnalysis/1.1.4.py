import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

input_data = np.load('data2D.npy')
np.random.seed(521)
randIndx = np.arange(len(input_data))
np.random.shuffle(randIndx)
input_data = input_data[randIndx]
trainData = input_data[:6667]
validData = input_data[6667:]

# data features
data_num = np.shape(input_data)[0]
data_dim = np.shape(input_data)[1]
tf.set_random_seed(521)

K=1000

sess = tf.InteractiveSession()

def Euclidean_dis(X, Y):
    #return the matrix containing the pairwise Euclidean distances
    
    X_b = tf.expand_dims(X, 1)
    result = X_b - Y
    result_square = tf.square(result)
    Euclidean_dis = tf.reduce_sum(result_square, 2)

    return Euclidean_dis
    
    
def graph(K,n):
    #define variables
    data = tf.placeholder(tf.float64, [None,data_dim], name='input_x')
    centorid =  tf.cast(tf.Variable(tf.random_normal([K, data_dim])),tf.float64)

    #calculate the distance, then assignment each point to the nearest centeroid
    Dis = Euclidean_dis(data,centorid)
    assignment = tf.arg_min(Dis,1)
    assignment = tf.cast(tf.one_hot(assignment,K),tf.float64)
    
    #calculate percentage of data points belonging to each of the K clusters
#    point_num = tf.expand_dims(tf.reduce_sum(assignment,0),1)
#    percentage = tf.cast(point_num / data_num,tf.float64)

    #difine Loss funtion
    U = tf.matmul(assignment,centorid)
    distance = data - U
    Loss = tf.reduce_sum(tf.square(distance))

    #define train algorithm
    optimizer = tf.train.AdamOptimizer(learning_rate=n,beta1=0.9,beta2=0.99,epsilon=1e-5)
    train = optimizer.minimize(loss=Loss)

    return train,Loss,centorid,data
    
    
def runMult(K,n):
    train_err = []
    valid_err = []
    # Build computation graph
    train,loss,centorid,data = graph(K,n)

    # Initialize session
    init = tf.initialize_all_variables()
    sess.run(init)

    #train model
    for i in np.arange(800):
        _, err, center= sess.run([train, loss, centorid], feed_dict={data:trainData})
        train_err.append(err)
        print('update_times:',i,'TRAIN_loss:',err)
        
        errValid = sess.run(loss, feed_dict= {data: validData})
        valid_err.append (errValid)
        print('update_times:',i,'VALID_loss:',errValid)
        
    #x = center[:, 0]
    #y = center[:, 1]
    #plt.plot(x,y,'*')
    plt.plot(train_err)
    plt.plot(valid_err)
    #plt.show()
    
    
runMult(K,0.01)

