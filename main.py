import tensorflow as tf
from konlpy.tag import Twitter
import numpy as np
import gensim
import time

def read_data(filename):    
    with open(filename, 'r',encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]        
        data = data[1:]
    return data 
    
train_data = read_data('data/ratings_train.txt')
# test_data = read_data('data/ratings_test.txt')
print(len(train_data))

twitter = Twitter() 

def tokenize(doc):

    return ['/'.join(t) for t in twitter.pos(doc, norm=True, stem=True)]

# tokens = [tokenize(row[1]) for row in train_data]

# model = gensim.models.Word2Vec(size=300, sg=1, alpha=0.025, min_alpha=0.025, seed=410)
# model.build_vocab(tokens)
# model.train(tokens, model.corpus_count, epochs=model.epochs)

# model.save('word2vec.model')

model = gensim.models.word2vec.Word2Vec.load('word2vec.model')

print(len(model.wv.vocab))

def convert2Vec(model, doc):  ## Convert corpus into vectors
    word_vec = []
    word_num = 0
    unk_num = 0
#     print('doc : ', doc)
    for sent in doc:
#         print('sent : ', sent)
        sub = []
        for word in sent:
            word_num += 1
            if(word in model.wv.vocab):
#                 print('word in sent')
                sub.append(model.wv[word])
            else:
                unk_num += 1
#                 print('word not in sent')
                sub.append(np.zeros(300)) ## used for OOV words
        word_vec.append(sub)
    print('word_num :', word_num, 'unk_num :', unk_num)
    return word_vec

tokens = [[tokenize(row[1]),int(row[2])] for row in train_data if tokenize(row[1]) != []]
tokens = np.array(tokens)
train_X = tokens[:,0]
train_Y = tokens[:,[-1]]
train_X = convert2Vec(model, train_X)
# train_Y = np.array([[y] for y in train_Y])
seq_length = [len(x) for x in train_X]
maxseq_length = max(seq_length)
print(maxseq_length)

def zero_padding(maxseq_length, seq):
#     print(len(seq))
    zero_pad = np.zeros((maxseq_length) * 300)
    seq_flat = np.reshape(seq, [-1])
#     print(len(seq_flat))
    zero_pad[0:len(seq) * 300] = seq_flat
#     print(len(zero_pad))
    return zero_pad
train_X = [zero_padding(maxseq_length, seq) for seq in train_X]
print(len(train_X[0]))

X = tf.placeholder(tf.float32, shape=[None, maxseq_length * 300])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([maxseq_length * 300, 1]), name='weight1')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
#                        tf.log(1 - hypothesis))
cost = -tf.reduce_mean(Y * tf.log(hypothesis + 1e-7) + (1 - Y) * tf.log(1 - hypothesis + 1e-7))
# cost_summ = tf.summary.scalar("cost", cost)

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
# acc_summ = tf.summary.scalar("accuracy", accuracy)

training_epochs = 10
Batch_size = 1
total_batch = int(len(train_X) / Batch_size)
save_path = './saved/model'
saver = tf.train.Saver()

with tf.Session() as sess:
#     merged_summary = tf.summary.merge_all()
#     writer = tf.summary.FileWriter("./logs/logistic")
#     writer.add_graph(sess.graph)  # Show the graph
    
#     start_time = time.time()

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):

        avg_loss = 0
        for step in range(total_batch):
            n = step * Batch_size
            train_batch_X = train_X[n : n + Batch_size]
            train_batch_Y = train_Y[n : n + Batch_size]
            
            cost_val, _ = sess.run([cost, train], feed_dict={X: train_batch_X, Y: train_batch_Y})
            avg_loss += cost_val
            acc = sess.run(accuracy , feed_dict={X: train_batch_X, Y: train_batch_Y})
            if step%1000 == 0:
                print('Batch : ', step + 1, '/', total_batch,
                  ', BCE in this minibatch: ', cost_val, 'accuracy: ', float(acc))

#         summary = sess.run(merged_summary, feed_dict={X: train_X, Y: train_Y})
#         writer.add_summary(summary, global_step=epoch)

#         acc = sess.run(accuracy , feed_dict={X: train_X, Y: train_Y})
        print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch))
#         print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch), ' accuracy: ', float(acc))
        saver.save(sess, save_path, epoch)
        
#     duration = time.time() - start_time
#     minute = int(duration / 60)
#     print('learning finish, minutes:', minute)
