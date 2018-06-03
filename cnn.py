import tensorflow as tf

class CNN():

    def __init__(self, batch_size, maxseq_length, embedding_size, learning_rate=0.001):
        self.batch_size = batch_size
        self.maxseq_length = maxseq_length
        self.embedding_size = embedding_size
        self.num_filters = 64
        self.learning_rate = learning_rate
        self._build_net()

    def _build_net(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.maxseq_length * self.embedding_size])
        self.input = tf.reshape(self.X, [-1, self.maxseq_length, self.embedding_size, 1])
        self.Y = tf.placeholder(tf.float32, shape=[None, 1])

        self.L2 = self.cnn_layer(2)
        self.L3 = self.cnn_layer(3)
        self.L4 = self.cnn_layer(4)

        self.outputs = tf.concat([self.L2, self.L3, self.L4], 1)
        self.hypothesis = tf.contrib.layers.fully_connected(self.outputs, 1, activation_fn=tf.sigmoid)

        self.cost = -tf.reduce_mean(self.Y * tf.log(self.hypothesis + 1e-7) + (1 - self.Y) * tf.log(1 - self.hypothesis + 1e-7))
        self.cost_summ = tf.summary.scalar('cost', self.cost)

        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        self.predicted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicted, self.Y), dtype=tf.float32))
        self.acc_summ = tf.summary.scalar('accuracy', self.accuracy)

    def cnn_layer(self, filter_size):
        W = tf.Variable(tf.random_normal([filter_size, self.embedding_size, 1, self.num_filters], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.num_filters]))
        L = tf.nn.conv2d(self.input, W, strides=[1, 1, 1, 1], padding='VALID')
        L = tf.nn.relu(tf.nn.bias_add(L, b))
        L = tf.nn.max_pool(L, ksize=[1, self.maxseq_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        L_flat = tf.reshape(L, [-1, self.num_filters])
        return L_flat
