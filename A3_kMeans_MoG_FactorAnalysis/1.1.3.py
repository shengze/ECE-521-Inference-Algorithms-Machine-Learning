import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

input_data = np.load('data2D.npy')

# data features
data_num = np.shape(input_data)[0]
data_dim = np.shape(input_data)[1]
tf.set_random_seed(521)

K=5

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
    assignment_onehot = tf.cast(tf.one_hot(assignment,K),tf.float64)
    
    #calculate percentage of data points belonging to each of the K clusters
    point_num = tf.expand_dims(tf.reduce_sum(assignment_onehot,0),1)
    percentage = tf.cast(point_num / data_num,tf.float64)
    
    #get class1
    #a = tf.constant([0.,0.],tf.float64)
    #class1pre = tf.not_equal(tf.expand_dims(assignment[:,0],1)*data, a)
    #class1 = tf.reshape(tf.gather(data, tf.where(class1pre[:,0])),[-1,2])

    #difine Loss funtion
    U = tf.matmul(assignment_onehot,centorid)
    distance = data - U
    Loss = tf.reduce_sum(tf.square(distance))

    #define train algorithm
    optimizer = tf.train.AdamOptimizer(learning_rate=n,beta1=0.9,beta2=0.99,epsilon=1e-5)
    train = optimizer.minimize(loss=Loss)

    return train,Loss,centorid,data, percentage, assignment
    
    
def runMult(K,n):
    train_err = []
    # Build computation graph
    train,loss,centorid,data, percentage, assignment = graph(K,n)

    # Initialize session
    init = tf.initialize_all_variables()
    sess.run(init)

    #train model
    for i in np.arange(800):
        _, err, center, percent, assign= sess.run([train, loss, centorid, percentage,assignment], feed_dict={data:input_data})
        train_err.append(err)
        print('update_times:',i,'loss:',err)
        
    x = input_data[:, 0]
    y = input_data[:, 1]

    assign=np.int32(assign)
    #for i in range(10000):
        #if assign[i] == 1:
            #plt.scatter(x[i],y[i],c='c')
        #if assign[i] ==0:
            #plt.scatter(x[i],y[i],c='r')
    #plt.plot(train_err)
    print('percentage:',percent)
    print(assign)
    plt.scatter(x, y, c=assign)
    plt.show()
    
    
runMult(K,0.1)

