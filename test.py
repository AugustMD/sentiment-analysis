import tensorflow as tf
import numpy as np
from preprocess import *

embedding_size = 300
batch_size = 32
maxseq_length = 100

test_data = read_data('data/test.txt')
test_data = np.array(test_data)
test_X = test_data[:,0]
test_Y = test_data[:,[-1]]

model = word2vec_load()

X = tf.placeholder(tf.float32, shape=[None, maxseq_length * embedding_size])
Y = tf.placeholder(tf.float32, shape=[None, 1])

with tf.Session() as sess:
    total_batch = int(len(test_X) / batch_size)
    save_path = './saved/model-49'

    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph(save_path + '.meta')
    saver.restore(sess, save_path)

    graph = tf.get_default_graph()

    W = graph.get_tensor_by_name('weight:0')
    b = graph.get_tensor_by_name('bias:0')

    hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    total_acc = 0
    for step in range(total_batch):
        n = step * batch_size
        test_batch_X = test_X[n: n + batch_size]
        test_batch_X = [tokenize(row) for row in test_batch_X]
        test_batch_X = [embedding(model, row) for row in test_batch_X]
        test_batch_X = [zero_padding(maxseq_length, row, embedding_size) for row in test_batch_X]
        test_batch_Y = test_Y[n: n + batch_size]

        acc = sess.run(accuracy, feed_dict={X: test_batch_X, Y: test_batch_Y})
        total_acc += acc * len(test_batch_X)
        print('Batch : ', step + 1, '/', total_batch, 'accuracy: ', float(acc))

    print('accuracy:', float(total_acc / len(test_X)))
