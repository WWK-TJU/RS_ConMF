# -*- coding:utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# https://blog.csdn.net/the_lastest/article/details/79093407 对tfidf了解
from operator import itemgetter
# https://blog.csdn.net/qq_22022063/article/details/79019294 对operator的说明
corpus = ['This is the first document kaa',
      'This is the second second document.',
      'And the third one.',
      'Is this the first?']
vectorizer = TfidfVectorizer(max_df=0.5, max_features=40)
#print(vectorizer)
vectorizer.fit(corpus)
vocab = vectorizer.vocabulary_ #变为一个词典
X_vocab = sorted(vocab.items(), key=itemgetter(1))
#print(vocab)
print(X_vocab)
baseline_vectorizer = CountVectorizer(vocabulary=vocab)#可以将词汇表转换成向量,CountVectorizer是每种词汇
X_base = baseline_vectorizer.fit_transform(corpus)
#print(baseline_vectorizer)
print(X_base)
# train = []
# print(X_base.shape)
# print(X_base.shape[0])
# print(X_base.shape[1])
# for i in range(X_base.shape[0]):
#     user_rating = X_base[i].nonzero()[1]
#     train.append((i, user_rating[0]))
#     #print(X_base[i])
#     #print(user_rating)
#     #print(user_rating[0])
# for j in range(4):
# 	item_rating = X_base.tocsc().T[j].nonzero()[1]
# 	print(item_rating)#[2] [0 1] [0 3] [0]
# 	train.append((item_rating[0], j))
# print(train)
# print(X_base.nonzero()[0]) #[0 0 0 1 1 2 2 2 3]
# print(X_base.nonzero()[1])
# print(set(X_base.nonzero()[0], X_base.nonzero()[1]))
#运行结果如下：
# TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
#         dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
#         lowercase=True, max_df=0.5, max_features=40, min_df=1,
#         ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True,
#         stop_words=None, strip_accents=None, sublinear_tf=False,
#         token_pattern='(?u)\\b\\w\\w+\\b', tokenizer=None, use_idf=True,
#         vocabulary=None)
# {'document': 1, 'first': 2, 'third': 5, 'and': 0, 'second': 4, 'one': 3}
# [('and', 0), ('document', 1), ('first', 2), ('one', 3), ('second', 4), ('third', 5)]
# CountVectorizer(analyzer='word', binary=False, decode_error='strict',
#         dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
#         lowercase=True, max_df=1.0, max_features=None, min_df=1,
#         ngram_range=(1, 1), preprocessor=None, stop_words=None,
#         strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
#         tokenizer=None,
#         vocabulary={'document': 1, 'first': 2, 'third': 5, 'and': 0, 'second': 4, 'one': 3})
#   (0, 1)	1
#   (0, 2)	1
#   (1, 1)	1
#   (1, 4)	2
#   (2, 0)	1
#   (2, 3)	1
#   (2, 5)	1
#   (3, 2)	1
# (4, 6)