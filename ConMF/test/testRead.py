import os
import numpy as np
import sys

def read_rating( path):
    results = []
    if os.path.isfile(path):
        raw_ratings = open(path, 'r')
    else:
        print("Path (preprocessed) is wrong!")
        sys.exit()
    index_list = []
    rating_list = []
    all_line = raw_ratings.read().splitlines()
    for line in all_line:
        tmp = line.split()
        print tmp[1]
        num_rating = int(tmp[0])
        # print  tmp[1]
        if num_rating > 0:
            tmp_i, tmp_r = zip(*(elem.split(":") for elem in tmp[1::]))
            # print tmp_i
            # print tmp_r
            index_list.append(np.array(tmp_i, dtype=int))
            rating_list.append(np.array(tmp_r, dtype=float))
        else:
            index_list.append(np.array([], dtype=int))
            rating_list.append(np.array([], dtype=float))

    # print index_list
    results.append(index_list)
    results.append(rating_list)

    return results

if __name__ == '__main__':
    path = "../data/movielens/preprocessed/movielens_1m/cf/0.2_1/test_user.dat"
    re = read_rating(path)
    # print re
