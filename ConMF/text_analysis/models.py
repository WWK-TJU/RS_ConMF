# -*-codinhg:utf-8-*-
'''
Created on Dec 8, 2015
@author: donghyun
'''

import keras
import numpy as np
from keras import Input
from keras.layers import GlobalMaxPooling1D
from keras.models import Model

np.random.seed(1337)

from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


class CNN_module():
    '''
    classdocs
    #https://www.cnblogs.com/shuibingyue/p/7300043.html　－－－－>详细的说明
    '''
    batch_size = 128
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5
    '''
    @nb_filters 
    Wjc提取cji（第i个单词的j号上下文特征），每篇文档词序列长度为l，共有l−ws+1个单词会被提取上下文特征，
    每个单词由nc种不同的共享权重W1c,...,Wjc,...,Wncc提取nc种特征，所以一篇文档，
    经卷积层提取出的上下文特征的shape是nc∗(l−ws+1)，nc相当于用于图像识别的CNN中特征图深度这一概念，
    这里特征图深度nc=nb_filters=100
    是感受眼个数也就是其最后文本的提取出的特征维度
    '''
    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters, init_W=None):

        self.max_len = max_len#词的个数
        max_features = vocab_size
        vanila_dimension = 200
        projection_dimension = output_dimesion #输出层的维度

        filter_lengths = [3, 4, 5]

        '''Embedding Layer'''
        inputs = Input(name='input', shape=(max_len,), dtype='int32')
        # 加载动态词向量还是静态词向量
        # Embeddinghttps://keras-cn.readthedocs.io/en/latest/layers/embedding_layer/

        if init_W is None:
            embed_x = Embedding(input_dim=max_features, output_dim=emb_dim, input_length=max_len, name='sentence_embeddings')(inputs)
        else:
            embed_x = Embedding(max_features, emb_dim, input_length=max_len, weights=[init_W / 20],name='sentence_embeddings')(inputs)
        conv_list = []
        '''Convolution Layer & Max Pooling Layer'''
        for i in filter_lengths:
            feature_map = Conv1D(nb_filters, kernel_size=i, activation="relu",name="conv_"+str(i))(embed_x)
            max_x = GlobalMaxPooling1D(name="max_pooling_"+str(i))(feature_map)
            conv_list.append( max_x )
        ## concat all conv output vector into a long vector  This is Flatten
        sentence_vector = keras.layers.concatenate(conv_list)
        '''Dropout Layer'''
        '''Dense就是常用的全连接层  vanila_dimension就是最后全连接层输出的维度'''
        fc1_x = Dense(vanila_dimension, activation='tanh',name='fully_connect')(sentence_vector)
        drop_x = Dropout(dropout_rate,name='dropout')(fc1_x)
        '''Projection Layer & Output Layer'''
        out = Dense(projection_dimension, activation='tanh',name='projection')(drop_x)

        # Output Layer
        self.model = Model(inputs=inputs, outputs=out)
        # compile主要完成损失函数和优化器的一些配置，是为训练服务的
        self.model.compile(optimizer='rmsprop',loss='mse',metrics=['mse'])

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    # def qualitative_CNN(self, vocab_size, emb_dim, max_len, nb_filters):
    #     self.max_len = max_len
    #     max_features = vocab_size
    #
    #     filter_lengths = [3, 4, 5]
    #     print("Build model...")
    #     self.qual_conv_set = {}
    #     '''Embedding Layer'''
    #     Input(name='input', shape=(max_len,), dtype=int)
    #
    #     self.qual_model.add_node(Embedding(max_features, emb_dim, input_length=max_len, weights=self.model.nodes['sentence_embeddings'].get_weights()),
    #                              name='sentence_embeddings', input='input')
    #
    #     '''Convolution Layer & Max Pooling Layer'''
    #     for i in filter_lengths:
    #         model_internal = Sequential()
    #         model_internal.add(
    #             Reshape(dims=(1, max_len, emb_dim), input_shape=(max_len, emb_dim)))
    #         self.qual_conv_set[i] = Convolution2D(nb_filters, i, emb_dim, activation="relu", weights=self.model.nodes[
    #                                               'unit_' + str(i)].layers[1].get_weights())
    #         model_internal.add(self.qual_conv_set[i])
    #         model_internal.add(MaxPooling2D(pool_size=(max_len - i + 1, 1)))
    #         model_internal.add(Flatten())
    #
    #         self.qual_model.add_node(
    #             model_internal, name='unit_' + str(i), input='sentence_embeddings')
    #         self.qual_model.add_output(
    #             name='output_' + str(i), input='unit_' + str(i))
    #     self.qual_model = Graph()
    #     self.qual_model.compile(
    #         'rmsprop', {'output_3': 'mse', 'output_4': 'mse', 'output_5': 'mse'})

    def train(self, X_train, V, item_weight, seed):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        np.random.seed(seed)
        # 随机打乱
        X_train = np.random.permutation(X_train)
        np.random.seed(seed)
        V = np.random.permutation(V)
        np.random.seed(seed)
        item_weight = np.random.permutation(item_weight)

        print("Train...CNN module")
        # print('x={X_train}, y={V},verbose=0, batch_size=self.batch_size, epochs=self.nb_epoch, sample_weight={item_weight}'.format(X_train=X_train,V=V,item_weight=item_weight ))
        history = self.model.fit(x=X_train, y=V,verbose=0, batch_size=self.batch_size, epochs=self.nb_epoch, sample_weight=item_weight)

        # cnn_loss_his = history.history['loss']
        # cmp_cnn_loss = sorted(cnn_loss_his)[::-1]
        # if cnn_loss_his != cmp_cnn_loss:
        #     self.nb_epoch = 1
        return history

    def get_projection_layer(self, X_train):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        Y = self.model.predict(X_train, batch_size=len(X_train))
        print('output:%s' %Y)
        return Y
