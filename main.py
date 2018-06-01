import tensorflow as tf
import numpy as np
from preprocess import *
import time

embedding_size = 300
training_epochs = 50
batch_size = 32
learning_rate = 0.001

train_data = read_data('data/train.txt')
train_data = np.array(train_data)
train_X = train_data[:,0]
train_Y = train_data[:,[-1]]

model = word2vec_load()

seq_length = [len(x) for x in train_X]
maxseq_length = max(seq_length)
maxseq_length = 100

X = tf.placeholder(tf.float32, shape=[None, maxseq_length * embedding_size])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([maxseq_length * embedding_size, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis + 1e-7) + (1 - Y) * tf.log(1 - hypothesis + 1e-7))
# cost_summ = tf.summary.scalar("cost", cost)

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
# acc_summ = tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    total_batch = int(len(train_X) / batch_size)
    save_path = './saved/model'
    saver = tf.train.Saver()

    #     merged_summary = tf.summary.merge_all()
    #     writer = tf.summary.FileWriter("./logs/logistic")
    #     writer.add_graph(sess.graph)  # Show the graph

    start_time = time.time()

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):

        avg_loss = 0
        for step in range(total_batch):
            n = step * batch_size
            train_batch_X = train_X[n: n + batch_size]
            train_batch_X = [tokenize(row) for row in train_batch_X]
            train_batch_X = [embedding(model, row) for row in train_batch_X]
            train_batch_X = [zero_padding(maxseq_length, row, embedding_size) for row in train_batch_X]
            train_batch_Y = train_Y[n: n + batch_size]

            cost_val, _ = sess.run([cost, train], feed_dict={X: train_batch_X, Y: train_batch_Y})
            avg_loss += cost_val
            acc = sess.run(accuracy, feed_dict={X: train_batch_X, Y: train_batch_Y})
            print('Batch : ', step + 1, '/', total_batch, '(epoch:', epoch, ')',
                  ', BCE in this minibatch: ', cost_val, 'accuracy: ', float(acc))

        print('epoch:', epoch, ' train_loss:', float(avg_loss / total_batch))

        #         summary = sess.run(merged_summary, feed_dict={X: train_X, Y: train_Y})
        #         writer.add_summary(summary, global_step=epoch)

        #         acc = sess.run(accuracy , feed_dict={X: train_X, Y: train_Y})
        #         print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch), ' accuracy: ', float(acc))
        saver.save(sess, save_path, epoch)

    duration = time.time() - start_time
    minute = int(duration / 60)
    print('learning finish, minutes:', minute)
