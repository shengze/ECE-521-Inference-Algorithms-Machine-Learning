import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

input_data = np.load('data2D.npy')
#x = input_data[:, 0]
#y = input_data[:, 1]

# data features
data_num = np.shape(input_data)[0]
data_dim = np.shape(input_data)[1]
np.random.seed(521)
tf.set_random_seed(521)
#print(data_num,data_dim)
#plt.plot(x,y,'*')
#plt.show()
K=3

sess = tf.InteractiveSession()

def Euclidean_dis(X, Y):
    '''
    return the matrix containing the pairwise Euclidean distances
    '''

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

    #recalculate the centeroid
    #point_num = tf.expand_dims(tf.reduce_sum(assignment,0),1)
    #U_total = tf.matmul(tf.transpose(assignment),data)
    #U = U_total / point_num

    #difine Loss funtion
    U = tf.matmul(assignment,centorid)
    distance = data - U
    Loss = tf.reduce_sum(tf.square(distance))

    #define train algorithm
    optimizer = tf.train.AdamOptimizer(learning_rate=n,beta1=0.9,beta2=0.99,epsilon=1e-5)
    train = optimizer.minimize(loss=Loss)

    #print('center:',sess.run(centorid))
    #print('data',sess.run(data))
    #print('center',sess.run(centorid))
    #print('assignment:',sess.run(assignment))
    #print('dis_center:',sess.run(distance))
    return train,Loss,centorid,data

def runMult(K,n):
    train_err = []
    # Build computation graph
    train,loss,centorid,data = graph(K,n)

    # Initialize session
    init = tf.initialize_all_variables()
    sess.run(init)

    #train model

    for i in np.arange(400):
        _, err, center= sess.run([train, loss, centorid], feed_dict={data:input_data})
        train_err.append(err)
        print('update_times:',i,'loss:',err)

    #x = center[:, 0]
    #y = center[:, 1]
    #plt.plot(x,y,'*')
    #plt.show()
    print('mu:',center)
    plt.plot(train_err)
    plt.show()
runMult(K,0.05)


