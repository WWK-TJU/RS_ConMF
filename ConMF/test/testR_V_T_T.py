# -*- coding:utf-8-*-
import numpy as np
from scipy.sparse.csr import csr_matrix

a = [(0, 680), (1, 409), (2, 605), (3, 3100),(6039, 2742),(3100,3)]
b = {}
d = {}
for i, j in a:
    if i in b:
        b[i].append(j)
        print b
    else:
        b[i] = [j]
    if j in d:
        d[j].append(i)
    else:
        d[j] = [i]

print b
print d