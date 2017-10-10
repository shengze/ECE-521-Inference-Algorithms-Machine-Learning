import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

with np.load("notMNIST.npz") as data:
    Data, Target = data["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx].reshape(-1, 784) / 255.
    Target = Target[randIndx]
    Target_mat = np.zeros([len(Data), 10])
    Target_mat[np.arange(len(Data)), Target] = 1
    Target = Target_mat
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]

sess = tf.InteractiveSession()

n_nodes_hl1 = 1000
n_classes = 10
batch_size = 500
N = 0.9
lamda = 0.0003

n_updates = []
train_err = []
train_accuracy = []
test_err = []
test_accuracy = []
valid_err = []
valid_accuracy = []

x = tf.placeholder(tf.float64, [None, 784], name='input_x')
y = tf.placeholder(tf.float64, name='target_y')
keep_prob = tf.placeholder(tf.float64)

def neural_network_model(d, nodes, dropout):

    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784, nodes], stddev=3./(nodes+batch_size))),
                      'biases': tf.Variable(0.0, [nodes,])}
    output_layer = {'weights': tf.Variable(tf.random_normal([nodes, n_classes], stddev=3./(nodes+n_classes))),
                    'biases': tf.Variable(0.0, [n_classes,])}

    hidden_layer_1['weights'] = tf.cast(hidden_layer_1['weights'], tf.float64)
    hidden_layer_1['biases'] = tf.cast(hidden_layer_1['biases'], tf.float64)
    output_layer['weights'] = tf.cast(output_layer['weights'], tf.float64)
    output_layer['biases'] = tf.cast(output_layer['biases'], tf.float64)

    l1 = tf.add(tf.matmul(d, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l1 = tf.nn.dropout(l1, dropout)

    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

    return output, hidden_layer_1['weights'], output_layer['weights']


# neural_network_model(testData, nodes)


def train_neural_network(x):
    prediction, W1, W = neural_network_model(x, n_nodes_hl1, keep_prob)
    cost = tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y) + lamda * (tf.reduce_mean(tf.square(W1)) + tf.reduce_mean(tf.square(W)))/2))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    number_epochs = 5

    init = tf.initialize_all_variables()
    sess.run(init)

    for epoch in range(number_epochs):

        idx = np.arange(0, len(trainData))
        np.random.shuffle(idx)
        for i in range(0, int(len(trainData) / batch_size)):
            batch1 = trainData[idx]
            batch2 = trainTarget[idx]
            batch1 = batch1[i * batch_size:(i + 1) * batch_size]
            batch2 = batch2[i * batch_size:(i + 1) * batch_size]
#            epoch_data = np.float64(epoch_data)
#            epoch_data = trainData.next_batch(batch_size)
#            epoch_target = trainTarget.next_batch(batch_size)
#            p, c, _ = sess.run([prediction, cost, optimizer], feed_dict = {x: batch1, y: batch2})

            _, c, p, weight1 = sess.run([optimizer, cost, prediction, W1], feed_dict={x: batch1, y: batch2, keep_prob: 0.5})

            print('Updates', epoch * int(len(trainData) / batch_size) + i, 'completed out of',number_epochs * int(len(trainData) / batch_size), 'loss:', c)

            if i == 0:


                train_err.append(c)
                n_updates.append(epoch*int(len(trainData) / batch_size)+i)

                correct = np.equal(np.argmax(p, 1), np.argmax(batch2, 1))
                accuracy = np.mean(np.float64(correct))
                train_accuracy.append(accuracy)

            if not ((int(15000 / batch_size) * epoch + i) % int(15000 / batch_size * number_epochs / 4))or epoch * int(len(trainData) / batch_size) + i == number_epochs * int(len(trainData) / batch_size)-1:
                weight1 = np.transpose(weight1)
                weight1 = np.reshape(weight1, (1000,28,28))
                for ax in xrange(1, 1001):
                    fig = plt.subplot(50,20,ax)
                    plt.imshow(weight1[ax - 1], cmap = plt.cm.gray)
                    plt.axis('off')
                plt.show()
'''
            #TestData
                test_e, p_test = sess.run([cost,prediction], feed_dict={x: testData, y: testTarget, keep_prob: 1})
                test_err.append(test_e)
                correct_test = np.equal(np.argmax(p_test, 1), np.argmax(testTarget, 1))
                accuracy_test = np.mean(np.float64(correct_test))
                test_accuracy.append(accuracy_test)

            #ValidData
                valid_e, p_valid = sess.run([cost,prediction], feed_dict={x: validData, y: validTarget, keep_prob: 1})
                valid_err.append(valid_e)
                correct_valid = np.equal(np.argmax(p_valid, 1), np.argmax(validTarget, 1))
                accuracy_valid = np.mean(np.float64(correct_valid))
                valid_accuracy.append(accuracy_valid)
'''




#    print('Accuracy:',accuracy.eval({x:validData, y:validTarget}))
#    print('Accuracy:',accuracy.eval({x:testData, y:testTarget}))
#    plt.plot(n_updates, train_err, '-')
#    plt.plot(n_updates, test_err, '-')
#    plt.plot(n_updates, valid_err, '-')
#    plt.plot(train_accuracy)
#    plt.plot(test_accuracy)
#    plt.plot(valid_accuracy)


train_neural_network(x)