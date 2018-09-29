from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from operator import itemgetter
corpus = ['This is the first document kaa',
      'This is the second second document.',
      'And the third one.',
      'Is this the first?']

vectorizer = TfidfVectorizer(max_df=0.5,max_features=40)
# print vectorizer
vectorizer.fit(corpus)
vocab = vectorizer.vocabulary_
X_vocab = sorted(vocab.items(),key=itemgetter(1))
# print vocab
# print X_vocab
baseline_vectorizer = CountVectorizer(vocabulary=vocab)
X_base = baseline_vectorizer.fit_transform(corpus).toarray()
# print baseline_vectorizer
print X_base
train = []
for i in range(X_base.shape[0]):
    user_rating = X_base[i].nonzero()[0]
    print user_rating
    train.append((i, user_rating))

print train

# for j in range(4):
# 	item_rating = X_base.tocsc().T[j].nonzero()[1]
# 	print(item_rating)#[2] [0 1] [0 3] [0]
# 	train.append((item_rating[0], j))
#
# print(X_base.nonzero()[0]) #[0 0 0 1 1 2 2 2 3]
# print(X_base.nonzero()[1])
# print(set(X_base.nonzero()[0], X_base.nonzero()[1]))