#/usr/bin/python2
# coding: utf-8
'''
Preprocessing.
Make training/test data and a set of vocabulary. 
'''
from __future__ import print_function
import numpy as np
import pickle
import codecs
import sys
sys.stdout = sys.__stdout__


class Hyperparams:
    '''Hyper parameters'''
    batch_size = 8
    embed_dim = 300
    seqlen = 50  # We will predict the next/current word based on the preceding 50 characters.

def load_char_vocab():
    vocab = "EU abcdefghijklmnopqrstuvwxyz0123456789-.,?!'" # E: Empty, U:Unknown
    char2idx = {char:idx for idx, char in enumerate(vocab)}
    idx2char = {idx:char for idx, char in enumerate(vocab)}
    print("load_word_vocab done")
    return char2idx, idx2char      

def create_word_vocab():
    from collections import Counter
    from itertools import chain
    
    words = codecs.open('data/en_wikinews.txt', 'r', 'utf-8').read().split()
    word2cnt = Counter(chain(words))
    vocab = ["<EMP>", "<UNK>"] + [word for word, cnt in word2cnt.items() if cnt > 50]
    word2idx = {word:idx for idx, word in enumerate(vocab)}
    idx2word = {idx:word for idx, word in enumerate(vocab)} 
    pickle.dump( (word2idx, idx2word), open("data/word_vocab.pkl", "wb") )

def load_word_vocab():
    word2idx, idx2word = pickle.load( open("data/word_vocab.pkl", "rb") )
    print("load_word_vocab done")
    return word2idx, idx2word
    
# def create_data():
#     char2idx, idx2char = load_char_vocab()
#     word2idx, idx2word = load_word_vocab()
#     lines = codecs.open('data/en_wikinews.txt', 'r', 'utf-8').read().splitlines()
#     xs, ys = [], [] # vectorized sentences
#     for line in lines:
#         x, y = [], []
#         for i, word in enumerate(line.split()):
#             x.append(2) # space
#             y.append(word2idx.get(word, 1))
#             for char in word:
#                 x.append(char2idx.get(char, 1))
#                 y.append(word2idx.get(word, 1))
#         if len(x) <= 1000: #zero pre-padding
#             xs.append([0] * (1000 - len(x)) + x)
#             ys.append([0] * (1000 - len(x)) + y)
#
#     # Convert to 2d-arrays
#     X = np.array(xs)
#     Y = np.array(ys)
#
#     print("X.shape =", X.shape, "\nY.shape =", Y.shape)
#     np.savez('data/train.npz', X=X, Y=Y)

def create_data():
    char2idx, idx2char = load_char_vocab()
    word2idx, idx2word = load_word_vocab()
    lines = codecs.open('data/en_wikinews.txt', 'r', 'utf-8').read().splitlines()
    xs, ys = [], []

    for line in lines:
        if not line.strip():  # 跳过空行
            continue
        x, y = [], []
        print("x, y  Done")
        for i, word in enumerate(line.split()):
            x.append(2)  # space
            y.append(word2idx.get(word, 1))
            for char in word:
                x.append(char2idx.get(char, 1))
                y.append(word2idx.get(word, 1))
        if len(x) > 0 and len(x) <= 1000:
            xs.append([0] * (1000 - len(x)) + x)
            ys.append([0] * (1000 - len(x)) + y)

    if not xs:
        raise ValueError("没有有效的训练数据，请检查 en_wikinews.txt 文件内容！")

    X = np.array(xs)
    Y = np.array(ys)
    np.savez('data/train.npz', X=X, Y=Y)
    print("X.shape =", X.shape, "\nY.shape =", Y.shape)


def load_train_data():
    X = np.load('data/train.npz')['X'][:-64]
    Y = np.load('data/train.npz')['Y'][:-64]
    return X, Y

def load_test_data():
    X = np.load('data/train.npz')['X'][-64:]
    Y = np.load('data/train.npz')['Y'][-64:]
    return X, Y

if __name__ == '__main__':
    create_word_vocab()
    create_data()
    data = np.load("data/train.npz")
    print("X shape:", data["X"].shape)
    print("Y shape:", data["Y"].shape)
    print("Done")