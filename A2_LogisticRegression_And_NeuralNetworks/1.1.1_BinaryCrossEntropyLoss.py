import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

with np.load("notMNIST.npz") as data:
    Data, Target = data ["images"], data ["labels"]
    posClass = 2
    negClass = 9
    dataIndx = (Target==posClass) + (Target==negClass)
    Data = Data[dataIndx].reshape(-1, 784) / 255
    Target = Target[dataIndx].reshape(-1, 1)
    Target[Target==posClass] = 1
    Target[Target==negClass] = 0
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600]
    testData, testTarget = Data[3600:], Target[3600:]

sess = tf.InteractiveSession()

lamda = 0.01

batch_size = 500

number_updates = []
train_err0 = []
train_err1 = []
train_err2 = []
train_accuracy = []
test_err = []
test_accuracy = []

def buildGraph(N,lamda):
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[784,1],stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float64, [None,784], name='input_x')
    y_target = tf.placeholder(tf.float64, [None,1], name='target_y')
    
    W = tf.cast(W, tf.float64)
    b = tf.cast(b, tf.float64)
    # Graph definition
    y_logits = tf.matmul(X, W) + b
    y_predicted = tf.sigmoid(y_logits)
    # Error definition
    sigmoidCrossEntropyError = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_logits, y_target), name='sigCroEntroError') + lamda * tf.reduce_sum(tf.square(W))/float(2), name='mean_error')
#    meanSquaredError = 1/2*tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target), reduction_indices=1, name='squared_error') + lamda * tf.reduce_mean(tf.matmul(tf.transpose(W),W)), name='mean_squared_error')
    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = N)
    train = optimizer.minimize(loss=sigmoidCrossEntropyError)
    return W, b, X, y_target, y_logits, y_predicted, sigmoidCrossEntropyError, train


def ClassAccuracy(Data, Target, currentW, currentb):
    S = tf.shape(Target)[0].eval()
    Target = tf.reshape(Target,[S]).eval()
    Data = tf.cast(Data,tf.float64)
    y_predicted_v = tf.sigmoid(tf.matmul(Data,currentW) + currentb)
    y_predicted_v = tf.reshape(y_predicted_v, [S]).eval()
    y = np.zeros(S)
    for i in range(S):
        if (y_predicted_v[i] >= 0.5):
            y[i] = 1
        else:
            y[i] = 0
        y[i] = abs(Target[i]-y[i])
    z = 1 - np.sum(y)/np.shape(Target)[0]
    return z


def runMult(N):
    train_err1 = []
    #Build computation graph
    W, b, X, y_target, y_logits, y_predicted, sigmoidCrossEntropyError, train = buildGraph(N,lamda)
    #Initialize session
    init = tf.initialize_all_variables()
    sess.run(init)
    # Training model
    for step in range(0,100):
        
        idx = np.arange(0,3500)
        np.random.shuffle(idx)
        for index in range(0,int(3500/batch_size)):
               
               batch1 = trainData[idx]
               batch2 = trainTarget[idx]
               batch1 = batch1[index*batch_size:(index+1)*batch_size]
               batch2 = batch2[index*batch_size:(index+1)*batch_size]
               _, err, currentW, currentb, yhat_x, yhat = sess.run([train, sigmoidCrossEntropyError, W, b, y_logits, y_predicted], feed_dict={X: batch1, y_target: batch2})

               if (1):
#               if not ((int(3500/batch_size)*step+index) % 5) or int(3500/batch_size)*step+index < 10 :
#                   plt.plot(yhat, 'o')
                   print("Iter: %3d, ERR-train: %4.2f"%(int(3500/batch_size)*step+index, err))
                   number_updates.append (int(3500/batch_size)*step+index)
                   train_err1.append (err)

                   train_accuracy.append (ClassAccuracy(batch1, batch2, currentW, currentb))
#                   print(train_accuracy)
#    plt.plot(number_updates, train_err1, '-')
#    plt.plot(number_updates, train_accuracy, '-')
#    plt.show()

# Testing model

                   errTest = sess.run(sigmoidCrossEntropyError, feed_dict= {X: testData, y_target: testTarget})
                   print("Iter: %3d, ERR-test: %4.2f"%(int(3500/batch_size)*step+index, errTest))
                   test_err.append (errTest)
                   test_accuracy.append (ClassAccuracy(testData, testTarget, currentW, currentb))
#                   print(test_accuracy)

#    plt.plot(number_updates, test_err, '-')

#    plt.plot(number_updates, test_accuracy, '-')
    return train_err1,test_err,train_accuracy, test_accuracy


# Tuning learing rate
#train_err0,_ = runMult(0.01)
#train_err1,_ = runMult(0.1)
train_err2,test_err,train_accuracy, test_accuracy = runMult(1)
#train_err.append(runMult(0.1))
#plt.plot(train_err0, label='learning rate = 0.01')
#plt.plot(train_err1, label='learning rate = 0.1')
#plt.plot(train_err2, label='learning rate = 1')
#legend = plt.legend(loc='upper right', shadow=True)
#plt.show()

# PLot error curve
#plt.plot(train_err2, label='training_curve')
#plt.plot(test_err, label='testing_curve')
#legend = plt.legend(loc='upper right', shadow= True)
#plt.show()

# Plot accuracy curve
plt.plot(train_accuracy, label='training_curve')
plt.plot(test_accuracy, label='testing_curve')
legend = plt.legend(loc='upper right', shadow= True)
plt.show()

print('best test classification accuracy', np.max(test_accuracy))
np.set_printoptions(precision=4)




