import tensorflow as tf

class Logistic():

    def __init__(self, maxseq_length, embedding_size, learning_rate=0.001):
        self.maxseq_length = maxseq_length
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self._build_net()

    def _build_net(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.maxseq_length * self.embedding_size])
        self.Y = tf.placeholder(tf.float32, shape=[None, 1])

        self.W = tf.Variable(tf.random_normal([self.maxseq_length * self.embedding_size, 1]), name='weight')
        self.b = tf.Variable(tf.random_normal([1]), name='bias')

        self.hypothesis = tf.sigmoid(tf.matmul(self.X, self.W) + self.b)

        self.cost = -tf.reduce_mean(self.Y * tf.log(self.hypothesis + 1e-7) + (1 - self.Y) * tf.log(1 - self.hypothesis + 1e-7))
        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        self.predicted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicted, self.Y), dtype=tf.float32))
