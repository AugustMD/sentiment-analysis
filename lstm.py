import tensorflow as tf

class LSTM():

    def __init__(self, batch_size, maxseq_length, embedding_size, hidden_size, keep_prob, learning_rate=0.001):
        self.batch_size = batch_size
        self.maxseq_length = maxseq_length
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self._build_net()

    def _build_net(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.maxseq_length * self.embedding_size])
        self.input = tf.reshape(self.X, [-1, self.maxseq_length, self.embedding_size])
        self.Y = tf.placeholder(tf.float32, shape=[None, 1])

        self.fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size, state_is_tuple=True)
        self.fw_cell = tf.contrib.rnn.DropoutWrapper(self.fw_cell, output_keep_prob=self.keep_prob)

        self.bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size, state_is_tuple=True)
        self.bw_cell = tf.contrib.rnn.DropoutWrapper(self.fw_cell, output_keep_prob=self.keep_prob)

        self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, self.input, dtype=tf.float32)
        self.outputs = tf.concat(self.outputs, 2)

        self.outputs = tf.reshape(self.outputs, [-1, 2 * self.maxseq_length * self.hidden_size])
        self.hypothesis = tf.contrib.layers.fully_connected(self.outputs, 1, activation_fn=tf.sigmoid)

        self.cost = -tf.reduce_mean(self.Y * tf.log(self.hypothesis + 1e-7) + (1 - self.Y) * tf.log(1 - self.hypothesis + 1e-7))
        self.cost_summ = tf.summary.scalar('cost', self.cost)

        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        self.predicted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicted, self.Y), dtype=tf.float32))
        self.acc_summ = tf.summary.scalar('accuracy', self.accuracy)
