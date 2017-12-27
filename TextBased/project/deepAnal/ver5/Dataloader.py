# -*- coding: utf-8 -*-
from __future__ import division
import xlrd
import json
import numpy as np
import string
import os
import random
import cPickle as pickle
import csv

def getValueFromDict(dict, key, extraKey):
    if key in dict.keys():
        return dict[key]
    else:
        return dict[extraKey]
def padding(list_to_padiing, length, value):
    if len(list_to_padiing)>length:
        return list_to_padiing[0:length]
    elif len(list_to_padiing) < length:
        for i in range(length - len(list_to_padiing)):
            list_to_padiing.append(value)
        return list_to_padiing
    else:
        return list_to_padiing

class Dataloder2():
    def __init__(self, args):
        self.mode = args.mode
        self.word2vec_dir = args.word2vec_dir
        self.loss_weight = args.loss_weight
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.min_learning_rate = args.min_learning_rate
        #training or continue training
        if self.mode == '1' or self.mode == '3':
            self.data_path = args.train_data_path
            self.data_label_path = args.train_data_label_path
            self.batch_size = args.batch_size
        #test
        elif self.mode == '2':
            self.data_path = args.test_data_path
            self.data_label_path = args.test_data_label_path
            self.batch_size = 1
        #parse
        elif self.mode == '4':
            self.data_path = args.parse_data_path
            self.batch_size = 1

        self.num_steps = args.num_steps
        self.pointer = 0
        self.num_classes = 17


        self.loadWord2vec()
        if self.mode != '4':
            self.loadData()
            self.create_data_label()
        else:
            self.loadCsvData()
            # self.create_data()


    def loadWord2vec(self):
        dictionary_path = os.path.join(self.word2vec_dir, 'gameAndyuliaokudata_dictionary.pkl')
        embedding_matrix_path = os.path.join(self.word2vec_dir, 'gameAndyuliaokudata_embedding.npy')
        reverse_dictionay_path = os.path.join(self.word2vec_dir, 'gameAndyuliaokudata_reverse_dictionary.pkl')
        with open(dictionary_path, 'r') as f:
            self.dictionary = pickle.load(f)
        with open(reverse_dictionay_path, 'r') as f:
            self.reverse_dictionary = pickle.load(f)
        self.vocab_size = len(self.dictionary)
        self.embedding_matrix = np.load(embedding_matrix_path)
        self.emb_size = self.embedding_matrix.shape[1]

    def create_data_label(self):
        data = []
        labels = []
        weights = []
        data_length = []
        for i in range(len(self.raw_data)):
            row = self.raw_data[i]
            label_nopadding = self.raw_label[i]
            weight_nopadding = self.raw_weight[i]
            content = row['sentence']
            try:
                sentence_list = [getValueFromDict(self.dictionary, k , u'。') for _, k in enumerate(content)]
            except:
                # print '*********************************88'
                continue
            # sentence_list = [getValueFromDict(self.dictionary, k, '。') for _, k in enumerate(content)]
            # sentence_list = self.padding(sentence_list, 3) #３是句号
            data_length.append(min(len(sentence_list), self.num_steps))
            sentence_list = padding(sentence_list, self.num_steps, 3) #３是句号
            label = padding(label_nopadding, self.num_steps, 0)
            weight = padding(weight_nopadding, self.num_steps, 0.0)
            sentence_list = np.array(sentence_list)
            label = np.array(label)
            weight = np.array(weight)
            data.append(sentence_list)
            labels.append(label)
            weights.append(weight)
        self.data = data
        self.labels = labels
        self.data_length = data_length
        self.weights = weights
        self.nrows = len(self.data)
        self.num_batch = int(self.nrows/self.batch_size)
        self.lr_decay = (self.learning_rate - self.min_learning_rate)/(self.epochs)
        self.data_index = [i for i in range(self.nrows)]

    def create_one_data(self,row):
        data = []
        data_length = []
        id = row['id']
        content = row['sentence']
        content = content.decode('utf8')
        try:
            sentence_list = [getValueFromDict(self.dictionary, k, u'。') for _, k in enumerate(content)]
        except:
            # print '*********************************88'
            raise Exception
        # sentence_list = [getValueFromDict(self.dictionary, k, u'。') for _, k in enumerate(content)]
        data_length.append(min(self.num_steps, len(sentence_list)))
        sentence_list = padding(sentence_list, self.num_steps, 3)  # ３是句号
        sentence_list = np.array(sentence_list)
        data.append(sentence_list)
        data = np.array(data)
        data = data.astype(np.int32)
        data_length = np.array(data_length)
        data_length = data_length.astype(np.int32)
        return data, data_length

    def create_data(self):
        data = []
        data_length = []
        for row in self.raw_data:
            id = row['id']
            content = row['sentence']
            content = content.decode('utf8')
            try:
                sentence_list = [getValueFromDict(self.dictionary, k , u'。') for _, k in enumerate(content)]
            except:
                # print '*********************************88'
                continue
            # sentence_list = [getValueFromDict(self.dictionary, k, u'。') for _, k in enumerate(content)]
            data_length.append(min(self.num_steps, len(sentence_list)))
            sentence_list = padding(sentence_list, self.num_steps, 3) #３是句号
            sentence_list = np.array(sentence_list)
            data.append(sentence_list)
        self.data = data
        self.data_length = data_length
        self.nrows = len(self.data)
        self.num_batch = int(self.nrows/self.batch_size)
        self.data_index = [i for i in range(self.nrows)]
    def loadData(self):
        csv_reader = csv.reader(open(self.data_path, 'r'))
        raw_data = []
        raw_label = []
        raw_weight = []
        for row in csv_reader:
            if row:
                item = {}
                item['id'] = row[0]
                item['sentence'] = row[1].decode('utf8')
                item['theme'] = row[2].decode('utf8')
                raw_data.append(item)
        with open(self.data_label_path, 'r') as f:
            for line in f:
                line = line.strip('\n')
               	line = [int(i) for i in line.split(' ')]
                weight = [1.0 if i==0 else self.loss_weight for i in line]
                raw_label.append(line)
                raw_weight.append(weight)
        self.raw_data = raw_data
        self.raw_label = raw_label
        self.raw_weight = raw_weight
    def loadCsvData(self):
        raw_data = []
        test_data = csv.reader(open(self.data_path))
        for row in test_data:
            item = {}
            item['id'] = row[0]
            item['sentence'] = row[1]
            raw_data.append(item)
        self.raw_data = raw_data

    def next_batch(self):
        index_range = self.data_index[self.pointer: self.pointer + self.batch_size]
        x = [self.data[i] for i in index_range]
        y = [self.labels[i] for i in index_range]
        length = [self.data_length[i] for i in index_range]
        weight = [self.weights[i] for i in index_range]
        self.pointer += self.batch_size
        x = np.array(x)
        y = np.array(y)
        length = np.array(length)
        weight = np.array(weight)
        x = x.astype(np.float32)
        y = y.astype(np.int32)
        length = length.astype(np.int32)
        weight = weight.astype(np.float32)
        return x, y, length, weight
    def reset_pointer(self):
        random.shuffle(self.data_index)
        self.pointer = 0

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


    data_path = './Data/train_20000.csv'
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
    writeList('./Data/train_20000_label.txt', label_list)
if __name__ == '__main__':
    makeLabel()
