import numpy as np
import matplotlib.pyplot as plt
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
test_err1 = []
test_err2 = []
N1 = 0.01
N2 = 0.01
batch_size = 500

# Define ClassAccuracy
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

#Optimal Logistic Regression without Weight Decay

def buildCEGraph():
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[784,1],stddev= 0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float64, [None,784], name='input_x')
    y_target = tf.placeholder(tf.float64, [None,1], name='target_y')
    
    W = tf.cast(W, tf.float64)
    b = tf.cast(b, tf.float64)
    # Graph definition
    y_logits = tf.matmul(X, W) + b
    y_predicted = tf.sigmoid(y_logits)
    # Error definition
    sigmoidCrossEntropyError = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_logits, y_target), reduction_indices=1, name='sigCroEntroError'), name='mean_error')
#    meanSquaredError = tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target), reduction_indices=1, name='squared_error'), name='mean_squared_error')
    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = N2)
    train = optimizer.minimize(loss=sigmoidCrossEntropyError)
    return W, b, X, y_target, y_logits, y_predicted, sigmoidCrossEntropyError, train

def runCEMult():  
    #Build computation graph
    W, b, X, y_target, y_logits, y_predicted, sigmoidCrossEntropyError, train = buildCEGraph()
    #Initialize session
    init = tf.initialize_all_variables()
    sess.run(init)
    # Training model
    for step in range(0,100):
        
        idx = np.arange(0,3500)
        np.random.shuffle(idx)
        for index in range(0,7):
               
               batch1 = trainData[idx]
               batch2 = trainTarget[idx]
               batch1 = batch1[index*batch_size:(index+1)*batch_size]
               batch2 = batch2[index*batch_size:(index+1)*batch_size]
               _, err, currentW, currentb, yhat_x, yhat = sess.run([train, sigmoidCrossEntropyError, W, b, y_logits, y_predicted], feed_dict={X: batch1, y_target: batch2})
               errTest = sess.run(sigmoidCrossEntropyError, feed_dict={X: validData, y_target: validTarget})
               test_err1.append(errTest)
               if not ((int(3500/batch_size)*step+index) % 50) or int(3500/batch_size)*step+index < 10 :
                   print("Iter: %3d, ERR-train: %4.2f"%(int(3500/batch_size)*step+index, err))
    
    train_accuracy = ClassAccuracy(batch1, batch2, currentW, currentb)
    # Testing model

    test_accuracy = ClassAccuracy(testData, testTarget, currentW, currentb)
    # Validation model
    errValid = sess.run(sigmoidCrossEntropyError, feed_dict= {X: validData, y_target: validTarget})
    valid_accuracy = ClassAccuracy(validData, validTarget, currentW, currentb)

    print("train_error: %4.2f, train_accuracy: %4.2f, valid_error: %4.2f, valid_accuracy: %4.2f, test_error: %4.2f, test_accuracy: %4.2f"%(err, train_accuracy, errValid, valid_accuracy, errTest, test_accuracy))

#Least Squares Solution without Weight Decay

def buildSEGraph():
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[784,1],stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float64, [None,784], name='input_x')
    y_target = tf.placeholder(tf.float64, [None,1], name='target_y')
    
    W = tf.cast(W, tf.float64)
    b = tf.cast(b, tf.float64)
    # Graph definition
    y_logits = tf.matmul(X, W) + b
    y_predicted = y_logits
    #y_predicted = tf.sigmoid(y_logits)
    # Error definition
#    sigmoidCrossEntropyError = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_logits, y_target), reduction_indices=1, name='sigCroEntroError'), name='mean_error')
    meanSquaredError = tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target), reduction_indices= 1, name='squared_error'), name='mean_squared_error')
    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = N1)
    train = optimizer.minimize(loss=meanSquaredError)
    return W, b, X, y_target, y_logits, y_predicted, meanSquaredError, train

def runSEMult():  
    #Build computation graph
    W, b, X, y_target, y_logits, y_predicted, meanSquaredError, train = buildSEGraph()
    #Initialize session
    init = tf.initialize_all_variables()
    sess.run(init)
    # Training model
    for step in range(0,100):
        
        idx = np.arange(0,3500)
        np.random.shuffle(idx)
        for index in range(0,7):
               
               batch1 = trainData[idx]
               batch2 = trainTarget[idx]
               batch1 = batch1[index*batch_size:(index+1)*batch_size]
               batch2 = batch2[index*batch_size:(index+1)*batch_size]
               _, err, currentW, currentb, yhat_x, yhat = sess.run([train, meanSquaredError, W, b, y_logits, y_predicted], feed_dict={X: batch1, y_target: batch2})
               errTest = sess.run(meanSquaredError, feed_dict={X: validData, y_target: validTarget})
               test_err2.append(errTest)
               if not ((int(3500/batch_size)*step+index) % 50) or int(3500/batch_size)*step+index < 10 :
                   print("Iter: %3d, ERR-train: %4.2f"%(int(3500/batch_size)*step+index, err))

    train_accuracy = ClassAccuracy(batch1, batch2, currentW, currentb)
    # Testing model
    errTest = sess.run(meanSquaredError, feed_dict= {X: testData, y_target: testTarget})
    test_accuracy = ClassAccuracy(testData, testTarget, currentW, currentb)
    # Validation model
    errValid = sess.run(meanSquaredError, feed_dict= {X: validData, y_target: validTarget})
    valid_accuracy = ClassAccuracy(validData, validTarget, currentW, currentb)

    print("train_error: %4.2f, train_accuracy: %4.2f, valid_error: %4.2f, valid_accuracy: %4.2f, test_error: %4.2f, test_accuracy: %4.2f"%(err, train_accuracy, errValid, valid_accuracy, errTest, test_accuracy))
runSEMult()
runCEMult()
plt.plot(test_err1, label='Cross-entropy loss')
plt.plot(test_err2, label='squared-error loss')
legend= plt.legend(loc='upper right', shadow= True)
plt.show()


np.set_printoptions(precision=4)