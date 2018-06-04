import tensorflow as tf
import numpy as np
from preprocess import *
from logistic import Logistic
from dnn import DNN
from rnn import RNN
from lstm import LSTM
from cnn import CNN

# model_type = logistic / dnn / rnn / lstm / cnn
model_type = 'dnn'
test_epoch = 9
maxseq_length = 100
embedding_size = 300
batch_size = 32
keep_prob = 1.0

test_data = read_data('data/test.txt')
test_data = np.array(test_data)
test_X = test_data[:,0]
test_Y = test_data[:,[-1]]

word2vec = word2vec_load()

if model_type == 'logistic':
    model = Logistic(maxseq_length, embedding_size)
elif model_type == 'dnn':
    model = DNN(maxseq_length, embedding_size)
elif model_type == 'rnn':
    model = RNN(batch_size, maxseq_length, embedding_size)
elif model_type == 'lstm':
    model = LSTM(batch_size, maxseq_length, embedding_size, keep_prob)
elif model_type == 'cnn':
    model = CNN(batch_size, maxseq_length, embedding_size)

with tf.Session() as sess:
    total_batch = int(len(test_X) / batch_size)
    save_path = './saved/' + model_type + '/model-' + str(test_epoch)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, save_path)

    total_acc = 0
    for step in range(total_batch):
        n = step * batch_size
        test_batch_X = test_X[n: n + batch_size]
        test_batch_X = preprocessing(word2vec, maxseq_length, embedding_size, test_batch_X)
        test_batch_Y = test_Y[n: n + batch_size]

        acc = sess.run(model.accuracy, feed_dict={model.X: test_batch_X, model.Y: test_batch_Y})
        total_acc += acc * len(test_batch_X)
        print('Batch : ', step + 1, '/', total_batch, 'accuracy: ', float(acc))

    print('accuracy:', float(total_acc / len(test_X)))
