# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import xlrd
import os
import csv

def test():
    # data_path = './Data/sentiments_from_dashen.csv'
    # csv_reader = csv.DictReader(open(data_path, 'r'))
    # total = 0
    # sentiment_dict = {}
    # for row in csv_reader:
    #     sentiment = row['sentiments']
    #     count = row['count']
    #     count = int(count)
    #     total += count
    #     sentiment_dict[sentiment] = count
    # # print (total)
    #
    # sentiment_dict_sorted = sorted(sentiment_dict.items(), key=lambda item: item[1], reverse=False)
    # write_list = []
    # for item in sentiment_dict_sorted[::-1]:
    #     print (item[0], item[1])
    #     write_list.append([item[0], item[1]])
    # writeList('./Data/sentiments_sorted.txt', write_list)
    data_path = '/home/luofeng/PycharmProjects/DFCompetition/deepAnal/ver6/Data/res1207_toupiao_mostnum_66551.csv'
    csv_reader = csv.reader(open(data_path, 'r'))
    count_dict = {}
    for row in csv_reader:
        sentiments = row[3]
        sentiments = sentiments.split(';')
        sentiments.pop()
        for item in sentiments:
            if item in count_dict.keys():
                count_dict[item] = count_dict[item] + 1
            else:
                count_dict[item] = 1
    sentiment_dict_sorted = sorted(count_dict.items(), key=lambda item: item[1], reverse=False)
    write_list = []
    for item in sentiment_dict_sorted[::-1]:
        print (item[0], item[1])
        write_list.append([item[0], item[1]])
    writeList('./Data/sentiments_sorted_fromResult.txt', write_list)



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