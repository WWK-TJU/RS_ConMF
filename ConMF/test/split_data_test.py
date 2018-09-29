# -*-coding:utf-8 -*-
import numpy as np
import sys
import pickle as pickl

def split_data(ratio, R):
    print("Randomly splitting rating data into training set (%.1f) and test set (%.1f)..." % (1 - ratio, ratio))
    train = []
    # print R.shape #(6040,3544)这里的Ｒ是<class 'scipy.sparse.csr.csr_matrix'>

    for i in range(R.shape[0]):
        # 该行非零元素的列坐标,这可能有问题应该是[0], 可能会报错：tuple index out of range但是当R为csr_matrix不会出现这个问题
        user_rating = R[i].nonzero()[1]
        np.random.shuffle(user_rating)#打乱顺序
        train.append((i, user_rating[0]))#[(i,第一个元素)]
    # print train[(0, 680), (1, 409), (2, 605), (3, 3100),...(6039, 2742)]


    remain_item = set(range(R.shape[1])) - set(list(zip(*train))[1])#set集合的差集（未评论的item）

    for j in remain_item:
        item_rating = R.tocsc().T[j].nonzero()[1]#这必须得转置，这块是item　若不转置则为user
        np.random.shuffle(item_rating)
        train.append((item_rating[0], j))

    # print train[(0, 680), (1, 409), (2, 605),....(6037, 961), (6038, 799), (6039, 2742),
    # (4787, 4), (5999, 7), (3899, 11),...(194, 3541), (628, 3542)]
    rating_list = set(zip(R.nonzero()[0], R.nonzero()[1]))#非零元素的行，非零元素的列
    total_size = len(rating_list)
    remain_rating_list = list(rating_list - set(train))#剩下的评分
    np.random.shuffle(remain_rating_list)

    num_addition = int((1 - ratio) * total_size) - len(train)
    if num_addition < 0:
        print('this ratio cannot be handled')
        sys.exit()
    else:
        train.extend(remain_rating_list[:num_addition])
        tmp_test = remain_rating_list[num_addition:]
        np.random.shuffle(tmp_test)
        valid = tmp_test[::2]#取所有的值步长为２
        test = tmp_test[1::2]

        trainset_u_idx, trainset_i_idx = zip(*train)
        trainset_u_idx = set(trainset_u_idx)
        trainset_i_idx = set(trainset_i_idx)
        if len(trainset_u_idx) != R.shape[0] or len(trainset_i_idx) != R.shape[1]:
            print("Fatal error in split function. Check your data again or contact authors")
            sys.exit()

    print("Finish constructing training set and test set")
    print train
    return train, valid, test

if __name__ == '__main__':
    path = "../data/movielens/preprocessed/movielens_1m"
    R = pickl.load(open(path + "/ratings.all", "rb"))
    # print type(R)
    print set(range(R.shape[1]))

    print("Load preprocessed rating data - %s" % (path + "/ratings.all"))

    split_data(0.2,R)