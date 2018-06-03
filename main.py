import tensorflow as tf
import numpy as np
from preprocess import *
from logistic import Logistic
from dnn import DNN
from rnn import RNN
from lstm import LSTM
from cnn import CNN
import time

# model_type = logistic / dnn / rnn / lstm / cnn
model_type = 'lstm'
maxseq_length = 100
embedding_size = 300
training_epochs = 5
batch_size = 32
learning_rate = 0.001
keep_prob = 0.7

train_data = read_data('data/train.txt')
train_data = np.array(train_data)
train_X = train_data[:,0]
train_Y = train_data[:,[-1]]

word2vec = word2vec_load()

if model_type == 'logistic':
    model = Logistic(maxseq_length, embedding_size, learning_rate)
elif model_type == 'dnn':
    model = DNN(maxseq_length, embedding_size, learning_rate)
elif model_type == 'rnn':
    model = RNN(batch_size, maxseq_length, embedding_size, learning_rate)
elif model_type == 'lstm':
    model = LSTM(batch_size, maxseq_length, embedding_size, keep_prob, learning_rate)
elif model_type == 'cnn':
    model = CNN(batch_size, maxseq_length, embedding_size, learning_rate)

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/' + model_type)
    writer.add_graph(sess.graph)

    total_batch = int(len(train_X) / batch_size)
    save_path = './saved/' + model_type + '/model'
    saver = tf.train.Saver(max_to_keep=None)

    start_time = time.time()

    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):

        avg_loss = 0
        for step in range(total_batch):
            n = step * batch_size
            train_batch_X = train_X[n: n + batch_size]
            train_batch_X = preprocessing(word2vec, maxseq_length, embedding_size, train_batch_X)
            train_batch_Y = train_Y[n: n + batch_size]

            cost_val, _ = sess.run([model.cost, model.train], feed_dict={model.X: train_batch_X, model.Y: train_batch_Y})
            avg_loss += cost_val
            acc = sess.run(model.accuracy, feed_dict={model.X: train_batch_X, model.Y: train_batch_Y})
            print('Batch : ', step + 1, '/', total_batch, '(epoch:', epoch, ')',
                  ', BCE in this minibatch: ', cost_val, 'accuracy: ', float(acc))

            summary = sess.run(merged_summary, feed_dict={model.X: train_batch_X, model.Y: train_batch_Y})
            writer.add_summary(summary, global_step=epoch * total_batch + step)

        print('epoch:', epoch, ' train_loss:', float(avg_loss / total_batch))
        saver.save(sess, save_path, epoch)

    duration = time.time() - start_time
    minute = int(duration / 60)
    print('learning finish, minutes:', minute)
