# -*- coding: UTF-8 -*-
from __future__ import division
import time
import argparse
import cPickle as pickle
from collections import Counter
from parse_file import method_2017_11_25
import csv
import os

def readTxt(path):
    res = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            line = [int(i) for i in line.split(' ')]
            res.append(line)
    return res

def getVoteResultMostnum(votes):
    result = []
    for item in votes:
        c = Counter(item)
        result.append(c.most_common(1)[0][0])
    return result
def getVoteResultOnlySame(votes):
    result = []
    for item in votes:
        a = item[0]
        flag = True
        for i in item:
            if i != a:
                flag = False
                break
        if flag:
            result.append(a)
        else:
            result.append(0)
    return result

def main():
    dir = '/home/luofeng/PycharmProjects/DFCompetition (复件)/deepAnal/ver6/Data/fortoupiao'
    csvfile = 'taiyi_semi_test.csv'
    file_list = ['predict_list_ver6.txt' ,'predict_list_ver6_2.txt' ,'predict_list_ver7.txt', 'predict_list_ver7_2.txt', 'predict_list_ver7_CRF.txt']
    res = [readTxt(os.path.join(dir, i)) for i in file_list]
    csv_reader = csv.reader(open(os.path.join(dir,csvfile), 'r'))

    save_file_path = 'result.csv'
    head = ['row_id', 'content', 'theme', 'sentiment_word', 'sentiment_anls']
    csvFile = open(os.path.join(save_file_path), 'w')
    writer = csv.writer(csvFile)
    writer.writerow(head)

    i = 0
    for row in csv_reader:
        id = row[0]
        if id == '424':
            aaa = 1
        content = row[1].decode('utf8')
        content_list = [c for _, c in enumerate(content)]
        tt = [res[k][i] for k in range(len(res))]
        votes = []
        for k in range(len(tt[0])):
            a = []
            for j in range(len(tt)):
                a.append(tt[j][k])
            votes.append(a)
        result = getVoteResultMostnum(votes)
        # result = getVoteResultOnlySame(votes)
        themes, sentiments, tags = method_2017_11_25(content, content_list, result)
        print id, content, themes, sentiments, tags
        i += 1
        writer.writerow([str(id).encode('utf8'), str(content).encode('utf8'), str(themes).encode('utf8'),
                             str(sentiments).encode('utf8'), str(tags).encode('utf8')])
    csvFile.close()
if __name__ == '__main__':
    main()