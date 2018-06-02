import tensorflow as tf

class RNN():

    def __init__(self, batch_size, maxseq_length, embedding_size, learning_rate=0.001):
        self.batch_size = batch_size
        self.maxseq_length = maxseq_length
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.hidden_size = 64
        self._build_net()

    def _build_net(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.maxseq_length * self.embedding_size])
        self.input = tf.reshape(self.X, [-1, self.maxseq_length, self.embedding_size])
        self.Y = tf.placeholder(tf.float32, shape=[None, 1])

        self.cell = tf.contrib.rnn.BasicRNNCell(num_units=self.hidden_size)
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)
        self.outputs, self.states = tf.nn.dynamic_rnn(self.cell, self.input, initial_state=self.initial_state, dtype=tf.float32)

        # self.hypothesis = tf.sigmoid(self.outputs[:, -1])
        self.outputs = tf.reshape(self.outputs, [-1, self.maxseq_length * self.hidden_size])
        self.hypothesis = tf.contrib.layers.fully_connected(self.outputs, 1, activation_fn=tf.sigmoid)

        self.cost = -tf.reduce_mean(self.Y * tf.log(self.hypothesis + 1e-7) + (1 - self.Y) * tf.log(1 - self.hypothesis + 1e-7))
        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        self.predicted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicted, self.Y), dtype=tf.float32))
