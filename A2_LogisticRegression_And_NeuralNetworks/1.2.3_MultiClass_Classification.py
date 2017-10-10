import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

with np.load("notMNIST.npz") as data:
    Data, Target = data ["images"], data ["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx].reshape(-1, 784) / 255.
    Target = Target[randIndx]
    Target_mat = np.zeros([len(Data),10])
    Target_mat[np.arange(len(Data)),Target] = 1
    Target = Target_mat
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]

sess = tf.InteractiveSession()


lamda = 0.01
N = 0.05
batch_size = 500

number_updates = []
train_err1=[]
train_err2=[]
train_err3=[]
train_accuracy = []
test_err = []
test_accuracy = []

def buildGraph(lamda,n):
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[784,10]), name='weights')
#    b = tf.Variable(tf.truncated_normal(shape=[1,10]), name='biases')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float64, [None,784], name='input_x')
    y_target = tf.placeholder(tf.float64, [None,10], name='target_y')
    
    W = tf.cast(W, tf.float64)
    b = tf.cast(b, tf.float64)
    # Graph definition
    y_logits = tf.matmul(X, W) + b
#    y_logits = tf.reshape(y_logits, [-1,])
#    y_logits = tf.reshape(tf.matmul(X, W) + b, [-1,1,1])
    y_predicted = tf.nn.softmax(y_logits)
#    y_logits = tf.expand_dims(y_logits,[-1,])
    # Error definition
    softmaxCrossEntropyError = tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_logits,y_target), name='softmaxCroEntroError') + lamda * tf.reduce_mean(tf.matmul(tf.transpose(W),W))/2, name='mean_error')
#    softmaxCrossEntropyError = tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_logits, y_target), reduction_indices=1, name='softmaxCroEntroError') + lamda * tf.reduce_mean(tf.matmul(tf.transpose(W),W))/2, name='mean_error')
    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = n)
    train = optimizer.minimize(loss=softmaxCrossEntropyError)
    return W, b, X, y_target, y_logits, y_predicted, softmaxCrossEntropyError, train


def ClassAccuracy(Data, Target, currentW, currentb):
    y_predicted_v = tf.nn.softmax(tf.matmul(Data,currentW) + currentb)
    correct = tf.equal(tf.argmax(y_predicted_v,1),tf.argmax(Target,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float64)).eval()
    
    return accuracy


def runMult(n):
    train_err = []
    test_err = []
    #Build computation graph
    W, b, X, y_target, y_logits, y_predicted, softmaxCrossEntropyError, train = buildGraph(lamda,n)

    #Initialize session
    init = tf.initialize_all_variables()
    sess.run(init)
    # Training model
    for step in range(0,30):
        
        idx = np.arange(0,15000)
        np.random.shuffle(idx)
        for index in range(0,30):
               
               batch1 = trainData[idx]
               batch2 = trainTarget[idx]
               batch1 = batch1[index*batch_size:(index+1)*batch_size]
               batch2 = batch2[index*batch_size:(index+1)*batch_size]
               _, err, currentW, currentb, yhat_x, yhat = sess.run([train, softmaxCrossEntropyError, W, b, y_logits, y_predicted], feed_dict={X: batch1, y_target: batch2})

               if (1) :
                   
                   print("Iter: %3d, ERR-train: %4.2f"%(int(15000/batch_size)*step+index, err))
                   number_updates.append (int(15000/batch_size)*step+index)
                   train_err.append (err)
                   #train_accuracy.append (ClassAccuracy(batch1, batch2, currentW, currentb))
                   # Testing model
                   errTest = sess.run(softmaxCrossEntropyError, feed_dict= {X: testData, y_target: testTarget})
                   print("Iter: %3d, ERR-test: %4.2f"%(int(15000/batch_size)*step+index, errTest))
                   test_err.append (errTest)
                   test_accuracy.append (ClassAccuracy(testData, testTarget, currentW, currentb))
                   
    #plt.plot(number_updates, train_err, '-')
#    plt.plot(number_updates, test_err, '-')
#    plt.plot(number_updates, train_accuracy, '-')
#    plt.plot(number_updates, test_accuracy, '-')
    #plt.show()
    return train_err,test_err
#train_err1=runMult(0.001)
train_err2,test_err=runMult(0.01)
#train_err3=runMult(0.1)
#plt.plot(train_err1,label='learning rate = 0.001')
#plt.plot(train_err2,label='learning rate = 0.01')
#plt.plot(train_err3,label='learning rate = 0.1')
#le=plt.legend(loc='upper right', shadow=True)
#plt.show()

# Plot cross_entropy loss
#plt.plot(train_err2,label='training loss')
#plt.plot(test_err,label='testing loss')
#le=plt.legend(loc='upper right', shadow=True)
#plt.show()

# Plot accuracy
#plt.plot(train_accuracy,label='training accuracy')
#plt.plot(test_accuracy,label='testing accuracy')
#le=plt.legend(loc='lower right', shadow=True)
#plt.show()
print('best accuracy',np.max(test_accuracy))
np.set_printoptions(precision=4)
