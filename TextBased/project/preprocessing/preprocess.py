# -*- coding: utf-8 -*-
import xlrd
from openpyxl import Workbook
import json
import numpy as np
import string
import jieba.posseg as pseg
import os
import random
import cPickle as pickle
import csv

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

def makeLabel():
    '''
    other:0
    theme_b:1
    theme_m:2
    theme_e:3
    theme_s:4
    sentiment_b_1:5
    sentiment_m_1:6
    sentiment_e_1:7
    sentiment_s_1:8
    sentiment_b_-1:9
    sentiment_m_-1:10
    sentiment_e_-1:11
    sentiment_s_-1:12
    sentiment_b_0:13
    sentiment_m_0:14
    sentiment_e_0:15
    sentiment_s_0:16
    :return:
     label_list
    '''
    def getLabel(content_list, themes_list, sentiments_list, sentiments_anls_list):
        label = [0 for _ in range(len(content_list))]

        for i in range(len(themes_list)):
            theme = themes_list[i]
            if not theme == 'NULL' and len(theme) > 0:
                if theme in content:
                    if len(theme) == 1:
                        index = string.index(content, theme)
                        label[index] = 4
                    elif len(theme) == 2:
                        index = string.index(content, theme)
                        label[index] = 1
                        label[index + 1] = 3
                    else:
                        index = string.index(content, theme)
                        label[index] = 1
                        label[index + 1: index + len(theme) - 1] = [2 for _ in range(len(theme) - 2)]
                        label[index + len(theme) - 1] = 3
        for i in range(len(sentiments_list)):
            sentiment = sentiments_list[i]
            sentiment_anls = sentiments_anls_list[i]
            if not sentiment == 'NULL' and len(sentiment) > 0:
                if sentiment in content:
                    if len(sentiment) == 1:
                        index = string.index(content, sentiment)
                        if sentiment_anls == '1':
                            label[index] = 8
                        elif sentiment_anls == '-1':
                            label[index] = 12
                        elif sentiment_anls == '0':
                            label[index] = 16
                        else:
                            label[index] = 0
                    elif len(sentiment) == 2:
                        index = string.index(content, sentiment)
                        if sentiment_anls == '1':
                            label[index] = 5
                            label[index + 1] = 7
                        elif sentiment_anls == '-1':
                            label[index] = 9
                            label[index + 1] = 11
                        elif sentiment_anls == '0':
                            label[index] = 13
                            label[index + 1] = 15
                        else:
                            label[index] = 0
                            label[index + 1] = 0
                    else:
                        index = string.index(content, sentiment)
                        if sentiment_anls == '1':
                            label[index] = 5
                            label[index + 1: index + len(sentiment) - 1] = [6 for _ in range(len(sentiment) - 2)]
                            label[index + len(sentiment) - 1] = 7
                        elif sentiment_anls == '-1':
                            label[index] = 9
                            label[index + 1: index + len(sentiment) - 1] = [10 for _ in range(len(sentiment) - 2)]
                            label[index + len(sentiment) - 1] = 11
                        elif sentiment_anls == '0':
                            label[index] = 13
                            label[index + 1: index + len(sentiment) - 1] = [14 for _ in range(len(sentiment) - 2)]
                            label[index + len(sentiment) - 1] = 15
                        else:
                            label[index] = 0
                            label[index + 1: index + len(sentiment) - 1] = [0 for _ in range(len(sentiment) - 2)]
                            label[index + len(sentiment) - 1] = 0
        return label


    data_path = '/home/luofeng/PycharmProjects/DFCompetition (复件)/deepAnal/ver7/Data/train_new_40000.csv'
    csv_reader = csv.reader(open(data_path, 'r'))
    label_list = []
    for row in csv_reader:
        id = row[0]
        content = row[1].decode('utf8')
        themes = row[2].decode('utf8')
        sentiments = row[3].decode('utf8')
        sentiments_anls = row[4].decode('utf8')
        content_list = [e for _, e in enumerate(content)]
        themes_list = themes.split(';')[0:-1]
        sentiments_list = sentiments.split(';')[0:-1]
        sentiments_anls_list = sentiments_anls.split(';')[0:-1]
        label = getLabel(content_list, themes_list, sentiments_list, sentiments_anls_list)
        label_list.append(label)
        # print content, themes, sentiments, sentiments_anls, label
    writeList('./train_20000_label.txt', label_list)
def makeSpeech():
    data_path = '/home/luofeng/PycharmProjects/DFCompetition (复件)/deepAnal/ver7/Data/train_new_40000.csv'
    dictionary_path = '/home/luofeng/PycharmProjects/DFCompetition (复件)/deepAnal/ver7/utils/speech_dictionary.pkl'
    dictionary = pickle.load(open(dictionary_path, 'r'))
    csv_reader = csv.reader(open(data_path, 'r'))
    speech_list = []
    for row in csv_reader:
        content = row[1].decode('utf8')
        words = pseg.cut(content)
        l = []
        for word in words:
            ci = word.word
            tag = word.flag
            l.extend([dictionary[tag] for _ in range(len(ci))])
        print content, l
        speech_list.append(l)
    writeList('./train_new_40000_speech.txt', speech_list)
if __name__ == '__main__':
    makeLabel()
    makeSpeech()