import tensorflow as tf

class DNN():

    def __init__(self, maxseq_length, embedding_size, learning_rate=0.001):
        self.maxseq_length = maxseq_length
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.hidden_layer_size = 200
        self._build_net()

    def _build_net(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.maxseq_length * self.embedding_size])
        self.Y = tf.placeholder(tf.float32, shape=[None, 1])

        self.W1 = tf.Variable(tf.random_normal([self.maxseq_length * self.embedding_size, self.hidden_layer_size]), name='weight1')
        self.b1 = tf.Variable(tf.random_normal([self.hidden_layer_size]), name='bias1')
        self.L1 = tf.sigmoid(tf.matmul(self.X, self.W1) + self.b1, name='layer1')

        self.W2 = tf.Variable(tf.random_normal([self.hidden_layer_size, self.hidden_layer_size]), name='weight2')
        self.b2 = tf.Variable(tf.random_normal([self.hidden_layer_size]), name='bias2')
        self.L2 = tf.sigmoid(tf.matmul(self.L1, self.W2) + self.b2, name='layer2')

        self.W3 = tf.Variable(tf.random_normal([self.hidden_layer_size, self.hidden_layer_size]), name='weight3')
        self.b3 = tf.Variable(tf.random_normal([self.hidden_layer_size]), name='bias3')
        self.L3 = tf.sigmoid(tf.matmul(self.L2, self.W3) + self.b3, name='layer3')

        self.W4 = tf.Variable(tf.random_normal([self.hidden_layer_size, self.hidden_layer_size]), name='weight4')
        self.b4 = tf.Variable(tf.random_normal([self.hidden_layer_size]), name='bias4')
        self.L4 = tf.sigmoid(tf.matmul(self.L3, self.W4) + self.b4, name='layer4')

        self.W5 = tf.Variable(tf.random_normal([self.hidden_layer_size, self.hidden_layer_size]), name='weight5')
        self.b5 = tf.Variable(tf.random_normal([self.hidden_layer_size]), name='bias5')
        self.L5 = tf.sigmoid(tf.matmul(self.L4, self.W5) + self.b5, name='layer5')

        self.W6 = tf.Variable(tf.random_normal([self.hidden_layer_size, self.hidden_layer_size]), name='weight6')
        self.b6 = tf.Variable(tf.random_normal([self.hidden_layer_size]), name='bias6')
        self.L6 = tf.sigmoid(tf.matmul(self.L5, self.W6) + self.b6, name='layer6')

        self.W7 = tf.Variable(tf.random_normal([self.hidden_layer_size, self.hidden_layer_size]), name='weight7')
        self.b7 = tf.Variable(tf.random_normal([self.hidden_layer_size]), name='bias7')
        self.L7 = tf.sigmoid(tf.matmul(self.L6, self.W7) + self.b7, name='layer7')

        self.W8 = tf.Variable(tf.random_normal([self.hidden_layer_size, self.hidden_layer_size]), name='weight8')
        self.b8 = tf.Variable(tf.random_normal([self.hidden_layer_size]), name='bias8')
        self.L8 = tf.sigmoid(tf.matmul(self.L7, self.W8) + self.b8, name='layer8')

        self.W9 = tf.Variable(tf.random_normal([self.hidden_layer_size, self.hidden_layer_size]), name='weight9')
        self.b9 = tf.Variable(tf.random_normal([self.hidden_layer_size]), name='bias9')
        self.L9 = tf.sigmoid(tf.matmul(self.L8, self.W9) + self.b9, name='layer9')

        self.W10 = tf.Variable(tf.random_normal([self.hidden_layer_size, 1]), name='weight10')
        self.b10 = tf.Variable(tf.random_normal([1]), name='bias10')
        # self.L3 = tf.sigmoid(tf.matmul(self.L2, self.W3) + self.b3, name='layer3')

        self.hypothesis = tf.sigmoid(tf.matmul(self.L9, self.W10) + self.b10)

        self.cost = -tf.reduce_mean(self.Y * tf.log(self.hypothesis + 1e-7) + (1 - self.Y) * tf.log(1 - self.hypothesis + 1e-7))
        self.cost_summ = tf.summary.scalar('cost', self.cost)

        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        self.predicted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicted, self.Y), dtype=tf.float32))
        self.acc_summ = tf.summary.scalar('accuracy', self.accuracy)
