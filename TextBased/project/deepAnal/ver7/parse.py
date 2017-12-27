# -*- coding: UTF-8 -*-
from __future__ import division
import tensorflow as tf
import argparse
import os
import cPickle as pickle
import xlrd
from openpyxl import Workbook
import numpy as np
import json
import csv
from Dataloader import Dataloder
from model import Model
import traceback
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def loadWord2vec(dir):
    dictionary_path = os.path.join(dir, 'dictionary.json')
    embedding_path = os.path.join(dir, 'final_embeddings.npy')
    reverse_dictionay_path = os.path.join(dir, 'reverse_dictionary.json')
    with open(dictionary_path, 'r') as f:
        dictionary = json.load(f)
    with open(reverse_dictionay_path, 'r') as f:
        reverse_dictionary = json.load(f)
    embedding = np.load(embedding_path)
    feature_dim = embedding.shape[1]
    return embedding, dictionary, feature_dim


def padding(vector, num_steps, feature_dim):
    if len(vector) >= num_steps:
        vector = vector[0: num_steps]
    else:
        for _ in range(num_steps - len(vector)):
            vector.append(np.array([0. for _ in range(feature_dim)]))
    return vector

def create_one_data(row, dictionary, embedding, num_steps, feature_dim):
    sentence = row['content']
    try:
        sentence_list = [k for _, k in enumerate(sentence)]
    except:
        traceback.print_exc()
    wordVector = lambda x: embedding[dictionary[x]] if x in dictionary else embedding[dictionary['UNK']]
    sentence_vector = [wordVector(a) for a in sentence_list]
    sentence_vector = padding(sentence_vector, 60, feature_dim)
    sentence_vector = np.array(sentence_vector)
    sentence_vector = sentence_vector.reshape((1, num_steps, feature_dim))
    return sentence_vector

def parse():
    parse_file_path = '../../Data/taiyi_test_sentiment.xlsx'
    utils_dir = './utils'
    word2vec_dir = '../word2vec'
    save_file_path = '../../Data/taiyi_test_parsed.csv'
    num_steps = 60
    embedding, dictionary, feature_dim = loadWord2vec(word2vec_dir)
    raw_data = []
    xls_data = xlrd.open_workbook(parse_file_path)
    table = xls_data.sheets()[0]
    nrows = table.nrows  # 行数
    for i in range(1, nrows):
        row = table.row_values(i)
        if row:
            item = {}
            item['id'] = row[0]
            item['content'] = row[1]
            item['sentiment_word'] = row[2]
            item['sentiment_anls'] = row[3]
            raw_data.append(item)

    with open(os.path.join(utils_dir, 'config.pkl'), 'rb') as f:
        args = pickle.load(f)
    args.sampling = True
    model = Model(args)

    head = ['row_id', 'content', 'theme', 'sentiment_word', 'sentiment_anls']
    csvFile = open(save_file_path, 'w')
    writer = csv.writer(csvFile)
    writer.writerow(head)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for row in raw_data:
                row_id = row['id']
                content = row['content']
                sentiment_word = row['sentiment_word']
                sentiment_anls = row['sentiment_anls']
                theme = ''
                try:
                    sentence_vector = create_one_data(row, dictionary, embedding, num_steps, feature_dim)
                    sentence_list = [k for _, k in enumerate(content)]
                    result = model.predict_class(sess, sentence_vector)

                    def getchar(a):
                        if result[a] == 1:
                            return '[b]' + sentence_list[a]
                        elif result[a] == 2:
                            return '[m]' + sentence_list[a]
                        elif result[a] == 3:
                            return '[e]' + sentence_list[a]
                        elif result[a] == 4:
                            return '[s]' + sentence_list[a]
                        else:
                            return '*'

                    def getTheme(sentence_list, result, sentiment_word):
                        out = ''
                        for ii in range(min(len(result),len(sentence_list))):
                            if result[ii] == 1:
                                if ii+1<len(result) and result[ii+1]==3:
                                    word = sentence_list[ii] + sentence_list[ii+1] + ';'
                                    out = out + word
                                elif ii+2<len(result) and result[ii+1]==2 and result[ii+2]==3:
                                    word = sentence_list[ii] + sentence_list[ii+1] + sentence_list[ii+2] + ';'
                                    out = out + word
                                elif ii+3<len(result) and result[ii+1]==2 and result[ii+2]==2 and result[ii+3]==3:
                                    word = sentence_list[ii] + sentence_list[ii+1] + sentence_list[ii+2] + sentence_list[ii+3] +';'
                                    out = out + word
                            if result[ii] == 3:
                                if ii-1>0 and result[ii-1]==0:
                                    word = sentence_list[ii-1] + sentence_list[ii] + ';'
                                    out = out + word
                            if result[ii] == 4:
                                word = sentence_list[ii] + ';'
                                out = out + word
                        if len(out) == 0:
                            length = len(sentiment_word.split(';'))
                            out = 'NULL;'*(length-1)
                        return out

                    # out = [getchar(t) for t in range(min(len(sentence_list), len(result)))]
                    theme = getTheme(sentence_list, result, sentiment_word)
                    # out = ''.join(out)
                    print content + '%' + theme
                except:
                    theme = ''
                writer.writerow([str(row_id).encode('utf8'), str(content).encode('utf8'), str(theme).encode('utf8'), str(sentiment_word).encode('utf8'), str(sentiment_anls).encode('utf8')])
    csvFile.close()

if __name__ == '__main__':
    parse()