import tensorflow as tf
import numpy as np
from preprocess import *
from logistic import Logistic
from dnn import DNN
from rnn import RNN
from lstm import LSTM
from cnn import CNN

# model_type = logistic / dnn / rnn / lstm / cnn
model_type = 'lstm'
test_epoch = 0
maxseq_length = 100
embedding_size = 300
batch_size = 32
keep_prob = 1.0

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

def predict(sentence):
    with tf.Session() as sess:
        save_path = './saved/' + model_type + '/model-' + str(test_epoch)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, save_path)

        preprocessed_sentence = preprocessing(word2vec, maxseq_length, embedding_size, [sentence])

        hypothesis = sess.run(model.hypothesis, feed_dict={model.X: preprocessed_sentence})
        print(hypothesis)

        sentiment = sess.run(model.predicted, feed_dict={model.X: preprocessed_sentence})
        if sentiment == 1:
            print(sentence, "-> 긍정")
        else:
            print(sentence, "-> 부정")

while(1):
    sentence = input("문장을 입력해주세요 : ")
    predict(sentence)