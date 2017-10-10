import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


with np.load("tinymnist.npz") as data:
    trainData,trainTarget = data["x"],data["y"]
    validData,validTarget = data["x_valid"],data["y_valid"]
    testData,testTarget = data["x_test"],data["y_test"]
    
sess = tf.InteractiveSession()

y = np.zeros(np.shape(validData)[0])
accuracy = []
wList = []
bList = []
batch_size = 50
N = 0.2
z=[]
def buildGraph(lamda):
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[64,1],stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 64], name='input_x')
    y_target = tf.placeholder(tf.float32, [None, 1], name='target_y')
    
    # Graph definition
    y_predicted = tf.matmul(X,W) + b

    # Error definition0
    meanSquaredError = 1/2*tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target), reduction_indices=1, name='squared_error') + lamda * tf.reduce_mean(tf.matmul(tf.transpose(W),W)), name='mean_squared_error')
    
    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = N)
    train = optimizer.minimize(loss=meanSquaredError)
    return W, b, X, y_target, y_predicted, meanSquaredError, train
    
    
def runMult(lamda):
  
        
    #Build computation graph
    W, b, X, y_target, y_predicted, meanSquaredError, train = buildGraph(lamda)
   
    #Initialize session
    init = tf.initialize_all_variables()
    
    
    sess.run(init)
    
    initialW = sess.run(W)
    initialb = sess.run(b)
    
    #print("Initial weights: %s, initial bias: %.2f"%(initialW, initialb))
    # Training model
    coordinate_x = np.array([0,1,2,3,4,5,6,7,8,9,10,20]) 
    coordinate_y=np.array([])
   
    for step in range(0,20):
        
        idx = np.arange(0,700)
        np.random.shuffle(idx)
        for index in range(0,14):
               
           
               batch1 = trainData[idx]
               batch2 = trainTarget[idx]
               batch1 = batch1[index*batch_size:(index+1)*batch_size]
               batch2 = batch2[index*batch_size:(index+1)*batch_size]                
               _, err, currentW, currentb, yhat = sess.run([train, meanSquaredError, W, b, y_predicted], feed_dict={X: batch1, y_target: batch2})
#               W1 = np.float64(currentW)
#               b1 = np.float64(currentb)
#               y_predicted_v = np.matmul(validData,W1) + b1
#               for i in range(np.shape(y_predicted_v)[0]):
#                   if (y_predicted_v[i] > 0.5):
#                       y[i] = 1
#                   else:
#                       y[i] = 0
#                   y[i] = abs(validTarget[i]-y[i])
#               accuracy.append(1 - (np.sum(y)/100))
#               
#               
               #               z.append(err)
#               if not ((int(700/batch_size)*step+index) % 50) or int(700/batch_size)*step+index < 10 :
#                   print("Iter: %3d, MSE-train: %4.2f"%(int(700/batch_size)*step+index, err))
#                  
    # Testing model
    #errTest = sess.run(meanSquaredError, feed_dict= {X: testData, y_target: testTarget})
    #print("Final testing MSE: %.2f      lamda: %4.2f    batch_size: %3d"%(errTest, lamda, batch_size))
    #plt.plot(coordinate_x,coordinate_y)
    return currentW,currentb
    
np.set_printoptions(precision=4)


#
zl = [0]


for lamda in [0,0.0001,0.001,0.01,0.1,1]:
    W,b=runMult(lamda)
    W = np.float64(W)
    b = np.float64(b)
    y_predicted_v = np.matmul(validData,W) + b
    for i in range(np.shape(y_predicted_v)[0]):
        if (y_predicted_v[i] > 0.5):
            y[i] = 1
        else:
            y[i] = 0
        y[i] = abs(validTarget[i]-y[i])
    accuracy.append(1 - (np.sum(y)/100))

plt.plot(accuracy)
    
plt