# -*-coding:utf-8-*-

import codecs
import os
import sys
import pickle as pickl
import numpy as np
from operator import itemgetter
from scipy.sparse.csr import csr_matrix

# document.all 是一个字典，包括三个部分
# X_sequence :　其实和X_base 一样　这个是矩阵形式
#  X_vocab ：　8000维的词汇表
#  X_base　：　包含每一篇文章的词汇在8000维度的矩阵，词袋模型


def read_pretrained_word2vec(path, vocab, dim):
    if os.path.isfile(path):
        raw_word2vec = codecs.open(path, 'r', encoding='utf-8')
    else:
        print("Path (word2vec) is wrong!")
        sys.exit()

    word2vec_dic = {}
    all_line = raw_word2vec.read().splitlines()
    mean = np.zeros(dim)
    count = 0
    for line in all_line:
        tmp = line.split()
        _word = tmp[0]
        a = tmp[1]
        # b = tmp[1:]
        # print _word
        _vec = np.array(tmp[1:], dtype=float)
        # print _vec
        if _vec.shape[0] != dim:
            print(tmp)
            print('_vec.shape[0](%s) != dim(%s)' % (_vec.shape[0], dim))
            print("Mismatch the dimension of pre-trained word vector with word embedding dimension!")
            sys.exit()
        word2vec_dic[_word] = _vec
        mean = mean + _vec
        count = count + 1

    mean = mean / count

    W = np.zeros((len(vocab) + 1, dim))
    count = 0
    for _word, i in vocab:
        if _word in word2vec_dic:
            W[i + 1] = word2vec_dic[_word]
            count = count + 1
        else:
            W[i + 1] = np.random.normal(mean, 0.1, size=dim)

    print("%d words exist in the given pretrained model" % count)

    return W

if __name__ == '__main__':
    path="../data/movielens/preprocessed/movielens_1m"
    D_all = pickl.load(open(path + "/document.all", "rb"))
    print("Load preprocessed document data - %s" % (path + "/document.all"))

    # # print type(D_all) # is dict
    # #
    # print D_all['X_sequence'][3543]
    # # print type(D_all['X_sequence']) # is list
    # print np.shape( D_all['X_sequence'][3543]) # is (3544,)这个不是一维，而是第二维不确定
    #
    # print  "==================="
    # # print D_all['X_vocab']
    # # [(u'007', 0), (u'10', 1), (u'100', 2),...(u'zordon', 7996), (u'zorin', 7997), (u'zorro', 7998), (u'zyto', 7999)]
    # print type(D_all['X_vocab'])  # is list
    # print np.shape(D_all['X_vocab']) # is (8000,2)
    #
    # print  "==================="
    # # 这个就是３５４４个item 的词典
    # # print D_all['X_base']
    # print type(D_all['X_base'])  # is scipy.sparse.csr.csr_matrix
    # print np.shape(D_all['X_base'])  # is (3544,8000)

    path1 = "../data/glove/glove.6B.200d.txt"
    W = read_pretrained_word2vec(path1,D_all['X_vocab'],200)
    # print W