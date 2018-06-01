import numpy as np
from konlpy.tag import Twitter
import gensim

twitter = Twitter()

def read_data(filename):
    with open(filename, 'r',encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

def tokenize(doc):

    return ['/'.join(t) for t in twitter.pos(doc, norm=True, stem=True)]

def word2vec_train(tokens):
    model = gensim.models.Word2Vec(size=300, sg=1, alpha=0.025, min_alpha=0.025, seed=410)
    model.build_vocab(tokens)
    model.train(tokens, model.corpus_count, epochs=model.epochs)
    model.save('word2vec.model')

def word2vec_load():
    model = gensim.models.word2vec.Word2Vec.load('word2vec.model')
    return model

def embedding(model, sent):
    embedded = []
    for word in sent:
        if(word in model.wv.vocab):
            embedded.append(model.wv[word])
        else:
            embedded.append(np.zeros(300)) # used for OOV words
    return embedded

def zero_padding(max_length, seq, embedding_size):
    zero_pad = np.zeros((max_length) * embedding_size)
    seq_flat = np.reshape(seq, [-1])
    if(len(seq) > max_length):
        zero_pad[0:] = seq_flat[0:max_length * embedding_size]
    else:
        zero_pad[0:len(seq) * embedding_size] = seq_flat
    return zero_pad
