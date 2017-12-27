# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import gc
import math
import random
import pandas

# weights1 = [0.24, 0.14, 0.04]
weights2 = [100,90,80]
weights3 = [99,89,79]
# weights4 = [0.9, 0.45, 0.3]

# weights1 = [0.315, 0.215, 0.115]
# weights2 = [0.329, 0.229, 0.129]
# weights3 = [0.301, 0.201, 0.101]

# p1 = "submission.csv"
# p2 = "916result.csv"
p3 = 'final_mix912.csv'
p2 = 'result_919_1849_3day.csv'
p5 = "avg_920_1002.csv"

# f1 = open(p1)
f2 = open(p2)
f3 = open(p3)
# f4 = open(p4)
f_out = open(p5,'w')

# line1 = f1.readline()
line2 = f2.readline()
line3 = f3.readline()
# line4 = f4.readline()

while line2:
    predict = {}
    # array1 = line1.strip().split(',')
    array2 = line2.strip().split(',')
    array3 = line3.strip().split(',')
    # array4 = line4.strip().split(',')

    for i in range(3):
        # predict[array1[i+1]] = predict.get(array1[i+1],0)+weights1[i]
        predict[array2[i+1]] = predict.get(array2[i+1],0)+weights2[i]
        predict[array3[i+1]] = predict.get(array3[i+1],0)+weights3[i]
        # predict[array4[i+1]] = predict.get(array4[i+1],0)+weights4[i]
        # if array1[i+1][0:4] == 'fuck':
        #     predict[array1[i+1]]=0
        if array2[i+1] == '0':
            predict[array2[i+1]]=0
        if array3[i + 1] == '0':
            predict[array3[i + 1]] = 0
        # if array4[i + 1] == '0':
        #     predict[array4[i + 1]] = 0

    res = []
    for (id,prob) in sorted(predict.items(),key = lambda x:-x[1]):     # 这个地方有重复的
        res.append(id)

    res = res[:3]
    # line1 = f1.readline()
    line2 = f2.readline()
    line3 = f3.readline()
    # line4 = f4.readline()

    f_out.write(array2[0]+","+",".join(res)+"\n")
    # print >> f_out,array1[0]+","+",".join(res)

f_out.close()

