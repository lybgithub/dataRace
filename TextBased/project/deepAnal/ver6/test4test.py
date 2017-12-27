# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import xlrd
import os

def test():
    a = [1,2,3,4,5,6,7]
    a_pre = [3]
    a_pre = a[:-1]
    a_pre.insert(0, 3)
    a_pos = a[1:]
    a_pos.append(0)
    print (a)
    print (a_pre)
    print (a_pos)

def test_length_distribution():
    length_list = []
    with open('./Data/train_40000.csv') as f:
        for line in f:
            line = line.strip('\n')
            # print (line)
            content = line.split(',')[1]
            try:
                length = len([k for _, k in enumerate(content)])
                length_list.append(length)
            except:
                pass
    max_s = 200
    print("Percentage under {}: {}".format(max_s, len([w for w in length_list if w < max_s])/len(length_list)))
    print("length of length_list: {}".format(len(length_list)))
    plt.hist(length_list, bins=[i for i in xrange(max(length_list))])
    plt.show()

def writeList(filePath, listToWrite):
    # print listToWrite
    dir = os.path.split(filePath)[0]
    if not os.path.isdir(dir):
        os.makedirs(dir)
    if os.path.isfile(filePath):
        os.remove(filePath)
    f = open(filePath, 'w')
    for i in listToWrite:
        if isinstance(i, list):
            k = ' '.join([str(j) for j in i])
            f.write(k + "\n")
        elif isinstance(i, basestring):
            f.write(i + "\n")
    f.close()
def createData():
    data = []
    xls_data = xlrd.open_workbook('./Data/taiyi_train.xlsx')
    table = xls_data.sheets()[0]
    nrows = table.nrows
    for i in range(1, nrows):
        row = table.row_values(i)
        if row:
            id = int(row[0])
            content = row[1]
            theme = row[2]
            sentiment = row[3]
            sentiment_tag = row[4]
            try:
                out = str(id) + ',' + content + ',' + theme + ',' + sentiment + ',' + sentiment_tag
            except:
                out = str(id) + ',' + str(content) + ',' + theme + ',' + sentiment + ',' + sentiment_tag
            print (id)
            # out = id + ',' + content + ',' + theme + ',' + sentiment + ',' + sentiment_tag
            out = out.encode('utf8')
            data.append(out)
    with open('./Data/taiyi_semi_train.csv') as f:
        for line in f:
            line = line.strip('\r\n')
            data.append(line)
    writeList('./Data/train_40000.csv', data)
    writeList('./Data/train_30000.csv', data[:30000])
    writeList('./Data/train_10000.csv', data[30000:-1])
    aaa = 0
if __name__ == '__main__':
    # createData()
    test()