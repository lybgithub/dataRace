# -*- coding: UTF-8 -*-
from __future__ import division
import tensorflow as tf
import argparse
import os
import cPickle as pickle
import xlrd
from openpyxl import Workbook
import numpy as np
import string
import json
import csv
from Dataloader import Dataloder2
from model import Model
import jieba.posseg as pseg
import traceback
import math
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def txt2dict(path):
    result = dict()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.decode('utf8')
            # print line
            [word, tag] = line.split(' ')
            result[word] = tag
    # result = {c: i for i, c in enumerate(result)}
    return result
def findIndex(content, word):
    return string.index(content, word)
def average(l):
    return sum(l)/len(l)

def containCharacter(a):
    lower = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    upper = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    for i in lower:
        if i in a:
            return True
    for i in upper:
        if i in a:
            return True
    return False
def containDigit(a):
    digit = ['1','2','3','4','5','6','7','8','9','0']
    for i in digit:
        if i in a:
            return True

def getFenciTheme(content):
    fenci = []
    words = pseg.cut(content)
    for word in words:
        item = {}
        ci = word.word
        tag = word.flag
        if tag == 'n' or tag == 'nz' or tag == 'vn':
            index = findIndex(content, ci)
            item['word'] = ci
            item['index'] = index
            item['location'] = index + (len(ci)-1)/2
            fenci.append(item)
    return fenci

def getFenciDict(content):
    words = pseg.cut(content)
    ziidx2ciidx = {}
    ciidx2ci = {}
    ciidx2tag = {}
    ciidx = 0
    ziidx = 0
    for word in words:
        item = {}
        ci = word.word
        tag = word.flag
        ciidx2ci[ciidx] = ci
        ciidx2tag[ciidx] = tag
        for j in range(len(ci)):
            ziidx2ciidx[ziidx] = ciidx
            ziidx += 1
        ciidx += 1
    return ziidx2ciidx, ciidx2ci, ciidx2tag


class SentimentDictionary():
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.vocab = set(self.dictionary.keys())
        self.vocab_size = len(self.dictionary)
    def contains(self, item):
        return item in self.dictionary
    def getSentimentTag(self, item):
        return self.dictionary[item]
    def display(self):
        for k, v in self.dictionary.items():
            print 'word:{}, sentiment:{}'.format(k.encode('utf8'), v.encode('utf8'))

    def match(self, sentence):
        result = []
        for word in self.vocab:
            if word in sentence:
                word_dict = dict()
                index = string.index(sentence, word)
                tag = self.dictionary[word]
                word_dict['word'] = word
                word_dict['index'] = index
                word_dict['location'] = index + (len(word)-1)/2
                word_dict['tag'] = tag
                result.append(word_dict)
        if len(result) > 0:
            for i in range(len(result)):
                word = result[i]['word']
                index = result[i]['index']
                tag = result[i]['tag']
                if index - 1 >= 0 and sentence[index - 1] == u'不':
                    word = u'不' + word
                    index = index - 1
                    tag = '-1'
                    if word == u'不错':
                        tag = '1'
                    result[i]['word'] = word
                    result[i]['index'] = index
                    result[i]['location'] = index + (len(word)-1)/2
                    result[i]['tag'] = tag
                elif index - 1 >= 0 and sentence[index - 1] == u'太':
                    word = u'太' + word
                    index = index - 1
                    result[i]['word'] = word
                    result[i]['index'] = index
                    result[i]['location'] = index + (len(word)-1)/2
                elif index - 1 >= 0 and sentence[index - 1] == u'很':
                    word = u'很' + word
                    index = index - 1
                    result[i]['word'] = word
                    result[i]['index'] = index
                    result[i]['location'] = index + (len(word)-1)/2
        #
        result_filtered = []
        if len(result)>0:
            for i in range(0,len(result)):
                flag = True
                for j in range(0,len(result)):
                    if i == j:
                        continue
                    item_i = result[i]
                    item_j = result[j]
                    word_i = item_i['word']
                    index_i = item_i['index']
                    word_j = item_j['word']
                    index_j = item_j['index']
                    set_i = set([ii for ii in range(index_i, index_i + len(word_i))])
                    set_j = set([ii for ii in range(index_j, index_j + len(word_j))])
                    k = set_i&set_j
                    if k == set_i:
                        if k== set_j:
                            flag = True
                        else:
                            flag = False
                            break
                if flag:
                    result_filtered.append(result[i])

        if len(result_filtered)>0:
            i = 0
            while i < len(result_filtered):
                j = i+1
                while j < len(result_filtered):
                    item_i = result_filtered[i]
                    item_j = result_filtered[j]
                    word_i = item_i['word']
                    word_j = item_j['word']
                    if word_i == word_j:
                        result_filtered.pop(j)
                        j = j - 1
                    j += 1
                i += 1

        result_sorted = sorted(result_filtered, lambda x, y: cmp(x['index'], y['index']))
        return result_sorted

def getThemeAndSentiment(sentence_list, result):
    theme_list = []
    sentiment_list = []
    # for ii in range(min(len(result), len(sentence_list))):
    ii = 0
    while ii < min(len(result), len(sentence_list)):
        #主题词
        if result[ii] == 1:
            #1 3
            if ii + 1 < len(sentence_list) and ii + 1 < len(result) and result[ii + 1] == 3:
                word = sentence_list[ii] + sentence_list[ii + 1]
                item = {}
                item['word'] = word
                item['index'] = ii
                item['location'] = item['index'] + (len(word)-1)/2
                theme_list.append(item)
                ii += 2
                continue
            #1 2 3
            elif ii + 2 < len(sentence_list) and ii + 2 < len(result) and result[ii + 1] == 2 and result[ii + 2] == 3:
                word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2]
                item = {}
                item['word'] = word
                item['index'] = ii
                item['location'] = item['index'] + (len(word) - 1) / 2
                theme_list.append(item)
                ii += 3
                continue
            #1 2 0
            # elif ii + 2 < len(sentence_list) and ii + 2 < len(result) and result[ii + 1] == 2 and result[ii + 2] == 0:
            #     word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2]
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     theme_list.append(item)
            #     ii += 3
            #     continue
            #看结果找出来的特殊情况: 1 [0|12] 3
            # elif ii + 2 < len(sentence_list) and ii + 2 < len(result) and (result[ii + 1] == 0 or result[ii + 1] == 1 or result[ii + 1] == 12) and result[ii + 2] == 3:
            #     word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2]
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     theme_list.append(item)
            #     ii += 3
            #     continue
            #1 2 2 3
            elif ii + 3 < len(sentence_list) and ii + 3 < len(result) and result[ii + 1] == 2 and result[
                        ii + 2] == 2 and result[ii + 3] == 3:
                word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2] + sentence_list[ii + 3]
                item = {}
                item['word'] = word
                item['index'] = ii
                item['location'] = item['index'] + (len(word) - 1) / 2
                theme_list.append(item)
                ii += 4
                continue
        if result[ii] == 3:
            aaa = 1
            #0 3
            # if ii - 1 >= 0 and result[ii - 1] == 0:
            #     word = sentence_list[ii - 1] + sentence_list[ii]
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii-1
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     theme_list.append(item)
            #     ii += 1
            #     continue
            #0 2 3
            # elif ii - 2 >= 0 and result[ii - 1] == 2 and result[ii - 2] == 0 :
            #     word = sentence_list[ii - 2] + sentence_list[ii - 1] + sentence_list[ii]
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii-2
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     theme_list.append(item)
            #     ii += 1
            #     continue
        if result[ii] == 4:
            word = sentence_list[ii]
            item = {}
            item['word'] = word
            item['index'] = ii
            item['location'] = item['index'] + (len(word) - 1) / 2
            theme_list.append(item)
            ii += 1
            continue
        #正向情感词
        if result[ii] == 5:
            #5 7
            if ii + 1 < len(sentence_list) and ii + 1 < len(result) and result[ii + 1] == 7:
                word = sentence_list[ii] + sentence_list[ii + 1]
                item = {}
                item['word'] = word
                item['index'] = ii
                item['location'] = item['index'] + (len(word)-1)/2
                item['tag'] = '1'
                sentiment_list.append(item)
                ii += 2
                continue
            #5 6 7
            elif ii + 2 < len(sentence_list) and ii + 2 < len(result) and result[ii + 1] == 6 and result[ii + 2] == 7:
                word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2]
                item = {}
                item['word'] = word
                item['index'] = ii
                item['location'] = item['index'] + (len(word) - 1) / 2
                item['tag'] = '1'
                sentiment_list.append(item)
                ii += 3
                continue
            #5 6 0
            # elif ii + 2 < len(sentence_list) and ii + 2 < len(result) and result[ii + 1] == 6 and result[ii + 2] == 0:
            #     word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2]
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     item['tag'] = '1'
            #     sentiment_list.append(item)
            #     ii += 3
            #     continue
            # 看结果找出来的特殊情况: 5 [(0|5|7|8)] 7
            # elif ii + 2 < len(sentence_list) and ii + 2 < len(result) and result[ii+1] == 7 and \
            #         result[ii + 2] == 7:
            #     word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2]
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     item['tag'] = '1'
            #     sentiment_list.append(item)
            #     ii += 3
            #     continue
            # 看结果找出来的特殊情况: 5 10 7:这种情况基本是负面情感
            # elif ii + 2 < len(sentence_list) and ii + 2 < len(result) and result[ii + 1] == 10 and result[ii + 2] == 7:
            #     word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2]
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     item['tag'] = '-1'
            #     sentiment_list.append(item)
            #     ii += 3
            #     continue
            #5 6 6 7
            elif ii + 3 < len(sentence_list) and ii + 3 < len(result) and result[ii + 1] == 6 and result[
                        ii + 2] == 6 and result[ii + 3] == 7:
                word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2] + sentence_list[ii + 3]
                item = {}
                item['word'] = word
                item['index'] = ii
                item['location'] = item['index'] + (len(word) - 1) / 2
                item['tag'] = '1'
                sentiment_list.append(item)
                ii += 4
                continue
        if result[ii] == 7:
            aaa = 1
            #0 7
            # if ii - 1 >= 0 and result[ii - 1] == 0:
            #     word = sentence_list[ii - 1] + sentence_list[ii]
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii-1
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     item['tag'] = '1'
            #     sentiment_list.append(item)
            #     ii += 1
            #     continue
            #0 6 7
            # elif ii - 2 >= 0 and result[ii - 1] == 6 and result[ii - 2] == 0 :
            #     word = sentence_list[ii - 2] + sentence_list[ii - 1] + sentence_list[ii]
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii-1
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     item['tag'] = '1'
            #     sentiment_list.append(item)
            #     ii += 1
            #     continue
        if result[ii] == 8:
            word = sentence_list[ii]
            item = {}
            item['word'] = word
            item['index'] = ii
            item['location'] = item['index'] + (len(word) - 1) / 2
            item['tag'] = '1'
            sentiment_list.append(item)
            ii += 1
            continue

        #负面情感词
        if result[ii] == 9:
            #9 11
            if ii + 1 < len(sentence_list) and ii + 1 < len(result) and result[ii + 1] == 11:
                word = sentence_list[ii] + sentence_list[ii + 1]
                item = {}
                item['word'] = word
                item['index'] = ii
                item['location'] = item['index'] + (len(word) - 1) / 2
                item['tag'] = '-1'
                sentiment_list.append(item)
                ii += 2
                continue
            #9 10 11
            elif ii + 2 < len(sentence_list) and ii + 2 < len(result) and result[ii + 1] == 10 and result[ii + 2] == 11:
                word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2]
                item = {}
                item['word'] = word
                item['index'] = ii
                item['location'] = item['index'] + (len(word) - 1) / 2
                item['tag'] = '-1'
                sentiment_list.append(item)
                ii += 3
                continue
            #9 10 0
            # elif ii + 2 < len(sentence_list) and ii + 2 < len(result) and result[ii + 1] == 10 and result[ii + 2] == 0:
            #     word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2]
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     item['tag'] = '-1'
            #     sentiment_list.append(item)
            #     ii += 3
            #     continue
            # 看结果找出来的特殊情况: 9 [(5|6|7|9)] 11
            # elif ii + 2 < len(sentence_list) and ii + 2 < len(result) and result[ii+1] == 11 and \
            #         result[ii + 2] == 11:
            #     word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2]
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     item['tag'] = '-1'
            #     sentiment_list.append(item)
            #     ii += 3
            #     continue
            #9 10 10 11
            elif ii + 3 < len(sentence_list) and ii + 3 < len(result) and result[ii + 1] == 10 and result[
                        ii + 2] == 10 and result[ii + 3] == 11:
                word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2] + sentence_list[ii + 3]
                item = {}
                item['word'] = word
                item['index'] = ii
                item['location'] = item['index'] + (len(word) - 1) / 2
                item['tag'] = '-1'
                sentiment_list.append(item)
                ii += 4
                continue
        if result[ii] == 11:
            aaa = 1
            #0 11
            # if ii - 1 >= 0 and result[ii - 1] == 0:
            #     word = sentence_list[ii - 1] + sentence_list[ii]
            #     # out = out + word + ';'
            #     # out_list.append(word)
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii-1
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     item['tag'] = '-1'
            #     sentiment_list.append(item)
            #     ii += 1
            #     continue
            #0 10 11
            # elif ii - 2 >= 0 and result[ii - 1] == 10 and result[ii - 2] == 0:
            #     word = sentence_list[ii - 2] + sentence_list[ii - 1] + sentence_list[ii]
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii-2
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     item['tag'] = '-1'
            #     sentiment_list.append(item)
            #     ii += 1
            #     continue
        if result[ii] == 12:
            word = sentence_list[ii]
            item = {}
            item['word'] = word
            item['index'] = ii
            item['location'] = item['index'] + (len(word) - 1) / 2
            item['tag'] = '-1'
            sentiment_list.append(item)
            ii += 1
            continue
        #０情感词
        if result[ii] == 13:
            #13 15
            if ii + 1 < len(sentence_list) and ii + 1 < len(result) and result[ii + 1] == 15:
                word = sentence_list[ii] + sentence_list[ii + 1]
                item = {}
                item['word'] = word
                item['index'] = ii
                item['location'] = item['index'] + (len(word) - 1) / 2
                item['tag'] = '0'
                sentiment_list.append(item)
                ii += 2
                continue
            #13 14 15
            elif ii + 2 < len(sentence_list) and ii + 2 < len(result) and result[ii + 1] == 14 and result[ii + 2] == 15:
                word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2]
                item = {}
                item['word'] = word
                item['index'] = ii
                item['location'] = item['index'] + (len(word) - 1) / 2
                item['tag'] = '0'
                sentiment_list.append(item)
                ii += 3
                continue
            #13 14 0
            # elif ii + 2 < len(sentence_list) and ii + 2 < len(result) and result[ii + 1] == 14 and result[ii + 2] == 0:
            #     word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2]
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     item['tag'] = '0'
            #     sentiment_list.append(item)
            #     ii += 3
            #     continue
            #13 14 14 15
            elif ii + 3 < len(sentence_list) and ii + 3 < len(result) and result[ii + 1] == 14 and result[
                        ii + 2] == 14 and result[ii + 3] == 15:
                word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2] + sentence_list[ii + 3]
                item = {}
                item['word'] = word
                item['index'] = ii
                item['location'] = item['index'] + (len(word) - 1) / 2
                item['tag'] = '0'
                sentiment_list.append(item)
                ii += 4
                continue
        if result[ii] == 15:
            aaa = 1
            #0 15
            # if ii - 1 >= 0 and result[ii - 1] == 0:
            #     word = sentence_list[ii - 1] + sentence_list[ii]
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii-1
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     item['tag'] = '0'
            #     sentiment_list.append(item)
            #     ii += 1
            #     continue
            #0 14 15
            # elif ii - 2 >= 0 and result[ii - 1] == 14 and result[ii - 2] == 0:
            #     word = sentence_list[ii - 2] + sentence_list[ii - 1] + sentence_list[ii]
            #     item = {}
            #     item['word'] = word
            #     item['index'] = ii-2
            #     item['location'] = item['index'] + (len(word) - 1) / 2
            #     item['tag'] = '0'
            #     sentiment_list.append(item)
            #     ii += 1
            #     continue
        if result[ii] == 16:
            word = sentence_list[ii]
            item = {}
            item['word'] = word
            item['index'] = ii
            item['location'] = item['index'] + (len(word) - 1) / 2
            item['tag'] = '0'
            sentiment_list.append(item)
            ii += 1
            continue
        ii += 1
    return theme_list, sentiment_list
def getTheme(sentence_list, result):
    out = ''
    out_list = []
    for ii in range(min(len(result), len(sentence_list))):
        if result[ii] == 1:
            if ii + 1 < len(sentence_list) and ii + 1 < len(result) and result[ii + 1] == 3:
                word = sentence_list[ii] + sentence_list[ii + 1]
                out = out + word + ';'
                out_list.append(word)
            elif ii + 2 < len(sentence_list) and ii + 2 < len(result) and result[ii + 1] == 2 and result[ii + 2] == 3:
                word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2]
                out = out + word + ';'
                out_list.append(word)
            elif ii + 3 < len(sentence_list) and ii + 3 < len(result) and result[ii + 1] == 2 and result[ii + 2] == 2 and result[ii + 3] == 3:
                word = sentence_list[ii] + sentence_list[ii + 1] + sentence_list[ii + 2] + sentence_list[ii + 3]
                out = out + word + ';'
                out_list.append(word)
        if result[ii] == 3:
            if ii - 1 > 0 and result[ii - 1] == 0:
                word = sentence_list[ii - 1] + sentence_list[ii]
                out = out + word + ';'
                out_list.append(word)
        if result[ii] == 4:
            word = sentence_list[ii]
            out = out + word + ';'
            out_list.append(word)
    return out, out_list

def findThemeAndSentimentAndTag(theme_result, sentiment_result, windows, sentimentDictinary):
    def findClosedThemeIdx(sentimentItem, theme_list, windows):
        location_a = sentimentItem['location']
        sentiment_word = sentimentItem['word']
        closest_distance = 10
        word = -1
        for i in range(len(theme_list)):
            themeItem = theme_list[i]
            theme_word = themeItem['word']
            location_b = themeItem['location']
            if not sentiment_word in theme_word and not theme_word in sentiment_word and location_a - location_b <= windows \
                    and location_a - location_b > 0 and abs(location_a - location_b) < closest_distance \
                    and not containCharacter(theme_word) \
                    and not sentimentDictinary.contains(theme_word):
                word = i
                closest_distance = abs(location_a - location_b)
        if word == -1:
            closest_distance = 10
            for i in range(len(theme_list)):
                themeItem = theme_list[i]
                theme_word = themeItem['word']
                location_b = themeItem['location']
                if not sentiment_word in theme_word and not theme_word in sentiment_word and location_b - location_a <= 0.5 * windows \
                        and location_b - location_a > 0 and abs(location_a - location_b) < closest_distance \
                        and not containCharacter(theme_word) \
                        and not sentimentDictinary.contains(theme_word):
                    word = i
                    closest_distance = abs(location_a - location_b)
        return word

    def findClosedSentimentIdx(themeItem, sentiment_list, windows):
        location_a = themeItem['location']
        theme_word = themeItem['word']
        closest_distance = 10
        word = -1
        for i in range(len(sentiment_list)):
            sentimentItem = sentiment_list[i]
            sentiment_word = sentimentItem['word']
            location_b = sentimentItem['location']
            if not sentiment_word in theme_word and not theme_word in sentiment_word and location_b - location_a <= windows \
                    and location_b - location_a > 0 and abs(location_a - location_b) < closest_distance:
                word = i
                closest_distance = abs(location_a - location_b)
        if word == -1:
            closest_distance = 10
            for i in range(len(sentiment_list)):
                sentimentItem = sentiment_list[i]
                sentiment_word = sentimentItem['word']
                location_b = sentimentItem['location']
                if not sentiment_word in theme_word and not theme_word in sentiment_word and location_a - location_b <= 0.5 * windows \
                        and location_a - location_b > 0 and abs(location_a - location_b) < closest_distance:
                    word = i
                    closest_distance = abs(location_a - location_b)
        return word

    sentiments = ''
    themes = ''
    tags = ''

    sentiment2themeIdx = dict()
    theme2sentimentIdx = dict()
    for i in range(len(sentiment_result)):
        sentimentItem = sentiment_result[i]
        closest_theme_word_idx = findClosedThemeIdx(sentimentItem, theme_result, windows)
        sentiment2themeIdx[i] = closest_theme_word_idx
    for i in range(len(theme_result)):
        themeItem = theme_result[i]
        closest_sentiment_word_idx = findClosedSentimentIdx(themeItem, sentiment_result, windows)
        theme2sentimentIdx[i] = closest_sentiment_word_idx

    for i in range(len(sentiment2themeIdx)):
        sentimentItem = sentiment_result[i]
        if not containDigit(sentimentItem['word']):
            themeIdx = sentiment2themeIdx[i]
            if themeIdx == -1:
                themes += 'NULL' + ';'
                sentiments += sentimentItem['word'] + ';'
                if 'tag' in sentimentItem.keys():
                    tags += sentimentItem['tag'] + ';'
            else:
                themeItem = theme_result[themeIdx]
                sentimentIdx = theme2sentimentIdx[themeIdx]
                if sentimentIdx == i:
                    themes += themeItem['word'] + ';'
                    sentiments += sentimentItem['word'] + ';'
                    if 'tag' in sentimentItem.keys():
                        tags += sentimentItem['tag'] + ';'
                else:
                    themes += 'NULL' + ';'
                    sentiments += sentimentItem['word'] + ';'
                    if 'tag' in sentimentItem.keys():
                        tags += sentimentItem['tag'] + ';'
    return themes, sentiments, tags

def findThemeAndSentimentAndTagWithoutDictionary(theme_result, sentiment_result, windows1, windows2):
    def findClosedThemeIdx(sentimentItem, theme_list, windows):
        location_a = sentimentItem['location']
        sentiment_word = sentimentItem['word']
        closest_distance = 10
        word = -1
        for i in range(len(theme_list)):
            themeItem = theme_list[i]
            theme_word = themeItem['word']
            location_b = themeItem['location']
            if not sentiment_word in theme_word and not theme_word in sentiment_word and location_a - location_b <= windows \
                    and location_a - location_b > 0 and abs(location_a - location_b) < closest_distance \
                    and not containCharacter(theme_word):
                word = i
                closest_distance = abs(location_a - location_b)
        if word == -1:
            closest_distance = 10
            for i in range(len(theme_list)):
                themeItem = theme_list[i]
                theme_word = themeItem['word']
                location_b = themeItem['location']
                if not sentiment_word in theme_word and not theme_word in sentiment_word and location_b - location_a <= 0.5 * windows \
                        and location_b - location_a > 0 and abs(location_a - location_b) < closest_distance \
                        and not containCharacter(theme_word):
                    word = i
                    closest_distance = abs(location_a - location_b)
        return word

    def findClosedSentimentIdx(themeItem, sentiment_list, windows):
        location_a = themeItem['location']
        theme_word = themeItem['word']
        closest_distance = 10
        word = -1
        for i in range(len(sentiment_list)):
            sentimentItem = sentiment_list[i]
            sentiment_word = sentimentItem['word']
            location_b = sentimentItem['location']
            if not sentiment_word in theme_word and not theme_word in sentiment_word and location_b - location_a <= windows \
                    and location_b - location_a > 0 and abs(location_a - location_b) < closest_distance:
                word = i
                closest_distance = abs(location_a - location_b)
        if word == -1:
            closest_distance = 10
            for i in range(len(sentiment_list)):
                sentimentItem = sentiment_list[i]
                sentiment_word = sentimentItem['word']
                location_b = sentimentItem['location']
                if not sentiment_word in theme_word and not theme_word in sentiment_word and location_a - location_b <= 0.5 * windows \
                        and location_a - location_b > 0 and abs(location_a - location_b) < closest_distance:
                    word = i
                    closest_distance = abs(location_a - location_b)
        return word

    sentiments = ''
    themes = ''
    tags = ''

    sentiment2themeIdx = dict()
    theme2sentimentIdx = dict()
    for i in range(len(sentiment_result)):
        sentimentItem = sentiment_result[i]
        closest_theme_word_idx = findClosedThemeIdx(sentimentItem, theme_result, windows1)
        if closest_theme_word_idx == -1:
            closest_theme_word_idx = findClosedThemeIdx(sentimentItem, theme_result, windows2)
        sentiment2themeIdx[i] = closest_theme_word_idx
    for i in range(len(theme_result)):
        themeItem = theme_result[i]
        closest_sentiment_word_idx = findClosedSentimentIdx(themeItem, sentiment_result, windows1)
        if closest_sentiment_word_idx == -1:
            closest_sentiment_word_idx = findClosedSentimentIdx(themeItem, sentiment_result, windows2)
        theme2sentimentIdx[i] = closest_sentiment_word_idx

    for i in range(len(sentiment2themeIdx)):
        sentimentItem = sentiment_result[i]
        if not containDigit(sentimentItem['word']):
            themeIdx = sentiment2themeIdx[i]
            if themeIdx == -1:
                themes += 'NULL' + ';'
                sentiments += sentimentItem['word'] + ';'
                if 'tag' in sentimentItem.keys():
                    tags += sentimentItem['tag'] + ';'
            else:
                themeItem = theme_result[themeIdx]
                sentimentIdx = theme2sentimentIdx[themeIdx]
                if sentimentIdx == i:
                    themes += themeItem['word'] + ';'
                    sentiments += sentimentItem['word'] + ';'
                    if 'tag' in sentimentItem.keys():
                        tags += sentimentItem['tag'] + ';'
                else:
                    themes += 'NULL' + ';'
                    sentiments += sentimentItem['word'] + ';'
                    if 'tag' in sentimentItem.keys():
                        tags += sentimentItem['tag'] + ';'
    return themes, sentiments, tags

class Parser():
    def __init__(self,config_path):
        self.args = pickle.load(open(config_path, 'r'))
        # self.args.data_path = './Data/taiyi_test_debug.csv'
        self.args.data_path = './Data/taiyi_test.csv'
        self.dataloader = Dataloder2(isTraining=False, args=self.args)
        self.sess = tf.Session()
        self.model = Model(isTraining=False, args=self.args)
        self.saver = tf.train.Saver(tf.all_variables())
        self.ckpt = tf.train.get_checkpoint_state(self.args.save_dir)
        self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)


    def parse(self, content, x, length):
        content_list = [k for _, k in enumerate(content)]
        result = self.model.predict_class(self.sess, x, length)
        out, out_list = getTheme(content_list, result)
        result_list = []
        for word in out_list:
            item = {}
            item['word'] = word
            item['index'] = findIndex(content, word)
            item['location'] = item['index'] + (len(word)-1)/2
            result_list.append(item)
        return out, result_list
#模型加暴力，index粒度窗口
def method1():
    sentimentDict = txt2dict('./utils/sentimentDictionary_big.txt')
    sentimentDictinary = SentimentDictionary(sentimentDict)
    save_file_path = './Data/result_2017-11-09.csv'
    parser = Parser('./utils/config.pkl')
    windows = 10

    head = ['row_id', 'content', 'theme', 'sentiment_word', 'sentiment_anls']
    csvFile = open(save_file_path, 'w')
    writer = csv.writer(csvFile)
    writer.writerow(head)

    for i in range(parser.dataloader.nrows):
        if i == 8:
            aaa = 1
        row_id = parser.dataloader.raw_data[i]['id']
        content = parser.dataloader.raw_data[i]['sentence']
        content = content.decode('utf8')
        vector = parser.dataloader.data[i]
        vector = vector[np.newaxis, :]
        length = [parser.dataloader.data_length[i]]
        length = np.array(length)
        theme, theme_list = parser.parse(content, vector, length)
        # print theme
        # fenci_list = getFenciWord(content)
        # theme_list.extend(fenci_list)
        # theme_list = getFenciWord(content)
        sentitment_result = sentimentDictinary.match(content)
        aaa = 1

        def findClosedTheme(sentimentItem, theme_list, windows):
            index_a = sentimentItem['index'] + (len(sentimentItem['word']) - 1) / 2
            sentiment_word = sentimentItem['word']
            closest_distance = 10
            word = ''
            for themeItem in theme_list:
                theme_word = themeItem['word']
                index_b = themeItem['index'] + (len(theme_word) - 1) / 2
                if not sentiment_word in theme_word and not theme_word in sentiment_word and index_a - index_b <= windows \
                        and index_a - index_b > 0 and abs(
                            index_a - index_b) < closest_distance and not containCharacter(theme_word) \
                        and not sentimentDictinary.contains(theme_word):
                    word = theme_word
                    closest_distance = abs(index_a - index_b)
            if len(word) == 0:
                closest_distance = 10
                for themeItem in theme_list:
                    theme_word = themeItem['word']
                    index_b = themeItem['index'] + (len(theme_word) - 1) / 2
                    if not sentiment_word in theme_word and not theme_word in sentiment_word and index_b - index_a <= 0.5 * windows \
                            and index_b - index_a > 0 and abs(
                                index_a - index_b) < closest_distance and not containCharacter(theme_word) \
                            and not sentimentDictinary.contains(theme_word):
                        word = theme_word
                        closest_distance = abs(index_a - index_b)
            if len(word) > 0:
                return word
            else:
                return 'NULL'
                # return ''

        sentiments = ''
        sentiments_list = []
        themes = ''
        themes_list = []
        tags = ''
        tags_list = []
        for sentimentItem in sentitment_result:
            sentiment_word = sentimentItem['word']
            sentiment_tag = sentimentItem['tag']
            theme_word = findClosedTheme(sentimentItem, theme_list, windows)
            # if len(theme_word) > 0:
            # themes += theme_word + ';'
            themes_list.append(theme_word)
            # sentiments += sentiment_word + ';'
            sentiments_list.append(sentiment_word)
            # tags += sentiment_tag + ';'
            tags_list.append(sentiment_tag)

        null_num = 0
        for i in range(len(themes_list)):
            if themes_list[i] == 'NULL' and null_num < 2:
                null_num += 1
            elif themes_list[i] == 'NULL' and null_num >= 2:
                continue
            themes += themes_list[i] + ';'
            sentiments += sentiments_list[i] + ';'
            tags += tags_list[i] + ';'
        print row_id, content, '**', themes, '**', sentiments, '**', tags
        writer.writerow([str(row_id).encode('utf8'), str(content).encode('utf8'), str(themes).encode('utf8'),
                         str(sentiments).encode('utf8'), str(tags).encode('utf8')])
    csvFile.close()
#仅用分词，词粒度窗口
def method2():
    sentimentDict = txt2dict('./utils/sentimentDictionary_extend.txt')
    sentimentDictinary = SentimentDictionary(sentimentDict)
    save_file_path = './Data/result_2017-11-08.csv'
    parser = Parser('./utils/config.pkl')
    windows = 2

    head = ['row_id', 'content', 'theme', 'sentiment_word', 'sentiment_anls']
    csvFile = open(save_file_path, 'w')
    writer = csv.writer(csvFile)
    writer.writerow(head)
    for i in range(parser.dataloader.nrows):

        row_id = parser.dataloader.raw_data[i]['id']
        content = parser.dataloader.raw_data[i]['sentence']
        content = content.decode('utf8')
        vector = parser.dataloader.data[i]
        vector = vector[np.newaxis, :]
        theme, theme_list = parser.parse(content, vector)
        fenci_list = getFenciTheme(content)
        theme_list.extend(fenci_list)
        # theme_list = getFenciWord(content)

        ziidx2ciidx, ciidx2ci, ciidx2tag = getFenciDict(content)
        sentitment_result = sentimentDictinary.match(content)

        def findWindowsTheme(content, sentimentItem, ziidx2ciidx, ciidx2ci, ciidx2tag, windows):
            sentiment_word = sentimentItem['word']
            sentiment_index = sentimentItem['index']
            word = ''
            #前向
            ziidx = sentiment_index - 1
            if ziidx >= 0:
                ciidx = ziidx2ciidx[ziidx]
                for i in range(windows):
                    ciidx_ = ciidx - i
                    if ciidx_ >= 0:
                        if ciidx2tag[ciidx_] == 'n' or ciidx2tag[ciidx_] == 'vn' or ciidx2tag[ciidx_] == 'nz'\
                                or ciidx2ci[ciidx_] == u'做工' or ciidx2ci[ciidx_] == u'东西' or ciidx2ci[ciidx_] == u'感觉':
                            word_ = ciidx2ci[ciidx_]
                            if sentiment_word in word_:
                                continue
                            if word_ == u'有点':
                                continue
                            else:
                                word = word_
                                break
            #后向
            if len(word) == 0:
                back_window = int(windows / 2)
                sentiment_final_index = sentiment_index + len(sentiment_word) - 1
                ziidx = sentiment_final_index + 1
                if ziidx < len(content):
                    ciidx = ziidx2ciidx[ziidx]
                    for i in range(back_window):
                        ciidx_ = ciidx + i
                        if ciidx_ in ciidx2ci.keys():
                            if ciidx2tag[ciidx_] == 'n' or ciidx2tag[ciidx_] == 'vn' or ciidx2tag[ciidx_] == 'nz':
                                word_ = ciidx2ci[ciidx_]
                                if sentiment_word in word_:
                                    continue
                                else:
                                    word = word_
                                    break
            if len(word)>0:
                return word
            else:
                return 'NULL'

        sentiments = ''
        sentiments_list = []
        themes = ''
        themes_list = []
        tags = ''
        tags_list = []
        for sentimentItem in sentitment_result:
            sentiment_word = sentimentItem['word']
            sentiment_tag = sentimentItem['tag']
            sentiment_index = sentimentItem['index']
            flag = True
            for i in range(len(sentiment_word)):
                zi = sentiment_word[i]
                ziidx = sentiment_index + i
                if ciidx2tag[ziidx2ciidx[ziidx]] == 'n' or ciidx2tag[ziidx2ciidx[ziidx]] == 'nz' or ciidx2tag[ziidx2ciidx[ziidx]] == 'vn' or ciidx2tag[ziidx2ciidx[ziidx]] == 'v':
                    flag = False
                    break
            if flag:
                theme_word = findWindowsTheme(content, sentimentItem, ziidx2ciidx, ciidx2ci, ciidx2tag, windows)
                themes_list.append(theme_word)
                sentiments_list.append(sentiment_word)
                tags_list.append(sentiment_tag)

        null_num = 0
        for i in range(len(themes_list)):
            if themes_list[i] == 'NULL' and null_num < 3:
                null_num += 1
            elif themes_list[i] == 'NULL' and null_num >= 3:
                continue
            themes += themes_list[i] + ';'
            sentiments += sentiments_list[i] + ';'
            tags += tags_list[i] + ';'
        print row_id, content, '**', themes, '**', sentiments, '**', tags
        writer.writerow([row_id.encode('utf8'), content.encode('utf8'), themes.encode('utf8'),
                         sentiments.encode('utf8'), tags.encode('utf8')])
    csvFile.close()

def method3():
    sentimentDict = txt2dict('./utils/sentimentDictionary.txt')
    sentimentDictinary = SentimentDictionary(sentimentDict)
    save_file_path = './Data/result_2017-11-21.csv'
    parser = Parser('./utils/config.pkl')
    windows = 10

    head = ['row_id', 'content', 'theme', 'sentiment_word', 'sentiment_anls']
    csvFile = open(save_file_path, 'w')
    writer = csv.writer(csvFile)
    writer.writerow(head)

    for i in range(parser.dataloader.nrows):
        if i == 8:
            aaa = 1
        row_id = parser.dataloader.raw_data[i]['id']
        content = parser.dataloader.raw_data[i]['sentence']
        content = content.decode('utf8')
        vector = parser.dataloader.data[i]
        vector = vector[np.newaxis, :]
        length = [parser.dataloader.data_length[i]]
        length = np.array(length)
        theme, theme_result = parser.parse(content, vector, length)
        # theme_result = getFenciTheme(content)
        # print theme
        # fenci_list = getFenciWord(content)
        # theme_list.extend(fenci_list)
        sentitment_result = sentimentDictinary.match(content)

        # def findClosedThemeIdx(sentimentItem, theme_list, windows):
        #     location_a = sentimentItem['location']
        #     sentiment_word = sentimentItem['word']
        #     closest_distance = 10
        #     word = -1
        #     for i in range(len(theme_list)):
        #         themeItem = theme_list[i]
        #         theme_word = themeItem['word']
        #         location_b = themeItem['location']
        #         if not sentiment_word in theme_word and not theme_word in sentiment_word and location_a - location_b <= windows \
        #                 and location_a - location_b > 0 and abs(location_a - location_b) < closest_distance\
        #                 and not containCharacter(theme_word) \
        #                 and not sentimentDictinary.contains(theme_word):
        #             word = i
        #             closest_distance = abs(location_a - location_b)
        #     if word == -1:
        #         closest_distance = 10
        #         for i in range(len(theme_list)):
        #             themeItem = theme_list[i]
        #             theme_word = themeItem['word']
        #             location_b = themeItem['location']
        #             if not sentiment_word in theme_word and not theme_word in sentiment_word and location_b - location_a <= 0.5 * windows \
        #                     and location_b - location_a > 0 and abs(location_a - location_b) < closest_distance \
        #                     and not containCharacter(theme_word) \
        #                     and not sentimentDictinary.contains(theme_word):
        #                 word = i
        #                 closest_distance = abs(location_a - location_b)
        #     return word
        #
        # def findClosedSentimentIdx(themeItem, sentiment_list, windows):
        #     location_a = themeItem['location']
        #     theme_word = themeItem['word']
        #     closest_distance = 10
        #     word = -1
        #     for i in range(len(sentiment_list)):
        #         sentimentItem = sentiment_list[i]
        #         sentiment_word = sentimentItem['word']
        #         location_b = sentimentItem['location']
        #         if not sentiment_word in theme_word and not theme_word in sentiment_word and location_b - location_a <= windows \
        #                 and location_b - location_a > 0 and abs(location_a - location_b) < closest_distance:
        #             word = i
        #             closest_distance = abs(location_a - location_b)
        #     if word == -1:
        #         closest_distance = 10
        #         for i in range(len(sentiment_list)):
        #             sentimentItem = sentiment_list[i]
        #             sentiment_word = sentimentItem['word']
        #             location_b = sentimentItem['location']
        #             if not sentiment_word in theme_word and not theme_word in sentiment_word and location_a - location_b <= 0.5 * windows \
        #                     and location_a - location_b > 0 and abs(location_a - location_b) < closest_distance:
        #                 word = i
        #                 closest_distance = abs(location_a - location_b)
        #     return word
        #
        # sentiments = ''
        # themes = ''
        # tags = ''
        #
        # sentiment2themeIdx = dict()
        # theme2sentimentIdx = dict()
        # for i in range(len(sentitment_result)):
        #     sentimentItem = sentitment_result[i]
        #     sentiment_word = sentimentItem['word']
        #     sentiment_tag = sentimentItem['tag']
        #     closest_theme_word_idx = findClosedThemeIdx(sentimentItem, theme_result, windows)
        #     sentiment2themeIdx[i] = closest_theme_word_idx
        # for i in range(len(theme_result)):
        #     themeItem = theme_result[i]
        #     closest_sentiment_word_idx = findClosedSentimentIdx(themeItem, sentitment_result, windows)
        #     theme2sentimentIdx[i] = closest_sentiment_word_idx
        #
        # for i in range(len(sentiment2themeIdx)):
        #     sentimentItem = sentitment_result[i]
        #     themeIdx = sentiment2themeIdx[i]
        #     if themeIdx == -1:
        #         themes += 'NULL' + ';'
        #         sentiments += sentimentItem['word'] + ';'
        #         tags += sentimentItem['tag'] + ';'
        #     else:
        #         themeItem = theme_result[themeIdx]
        #         sentimentIdx = theme2sentimentIdx[themeIdx]
        #         if sentimentIdx == i:
        #             themes += themeItem['word'] + ';'
        #             sentiments += sentimentItem['word'] + ';'
        #             tags += sentimentItem['tag'] + ';'
        #         else:
        #             themes += 'NULL' + ';'
        #             sentiments += sentimentItem['word'] + ';'
        #             tags += sentimentItem['tag'] + ';'
        themes, sentiments, tags = findThemeAndSentimentAndTag(theme_result, sentitment_result, windows, sentimentDictinary)
        print row_id, content, '**', themes, '**', sentiments, '**', tags
        writer.writerow([str(row_id).encode('utf8'), str(content).encode('utf8'), str(themes).encode('utf8'),
                         str(sentiments).encode('utf8'), str(tags).encode('utf8')])
    csvFile.close()

def tryBiaoResult():
    test_file_path = './Data/taiyi_semi_test.csv'
    biao_result_path = './Data/ensemble1128_fromBiao_to_matchThemeAndSentiment.csv'
    save_file_path = './Data/result_to_biao.csv'

    head = ['row_id', 'content', 'theme', 'sentiment_word', 'sentiment_anls']
    csvFile = open(save_file_path, 'w')
    writer = csv.writer(csvFile)
    writer.writerow(head)
    test_data = csv.reader(open(test_file_path, 'r'))
    biao_lines = []
    test_lines = []
    with open(biao_result_path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            biao_lines.append(line.split(','))
    for line in test_data:
        test_lines.append(line)
    for i in range(len(biao_lines)):
        line = biao_lines[i]
        test_line = test_lines[i]
        id = line[0]
        content = test_line[1]
        themes = line[2]
        sentiments = line[3]
        # id, content, themes, sentiments = line.split(',')
        content = content.decode('utf8')
        themes = themes.decode('utf8')
        sentiments = sentiments.decode('utf8')
        themes_split = themes.split(';')
        sentiments_split = sentiments.split(';')
        themes_list = []
        sentiments_list = []
        for word in themes_split:
            word = word.strip()
            try:
                item = {}
                item['word'] = word
                item['index'] = findIndex(content, word)
                item['location'] = item['index'] + (len(word)-1)/2
                themes_list.append(item)
            except:
                continue
        for word in sentiments_split:
            try:
                word = word.strip()
                item = {}
                item['word'] = word
                item['index'] = findIndex(content, word)
                item['location'] = item['index'] + (len(word)-1)/2
                item['tag'] = '1'
                sentiments_list.append(item)
            except:
                continue
        themes_out, sentiments_out, tags_out = findThemeAndSentimentAndTagWithoutDictionary(themes_list, sentiments_list, 6, 10)
        print id,content, themes_out, sentiments_out, tags_out
        writer.writerow([str(id).encode('utf8'), str(content).encode('utf8'), str(themes_out).encode('utf8'),
                         str(sentiments_out).encode('utf8'), str(tags_out).encode('utf8')])
    csvFile.close()

def filterSentiments(sentiments_list):
    for i in range(len(sentiments_list)):
        sentiment = sentiments_list[i]['word']
        index = sentiments_list[i]['index']
        location = sentiments_list[i]['location']
        if sentiment[0]==u'很' and (len(sentiment)==3 or len(sentiment)==4):
            sentiments_list[i]['word'] = sentiment[1:]
            sentiments_list[i]['index'] = index + 1
            sentiments_list[i]['location'] = location + 0.5
        else:
            pass
    return sentiments_list

def method_2017_11_25(content, content_list, model_result):
    windows1 = 6
    windows2 = 10
    theme_list, sentiment_list = getThemeAndSentiment(content_list, model_result)
    sentiment_list = filterSentiments(sentiment_list)
    themes, sentiments, tags = findThemeAndSentimentAndTagWithoutDictionary(theme_list, sentiment_list, windows1, windows2)
    return themes, sentiments, tags

if __name__ == '__main__':
    # method1()
    # method3()
    method_2017_11_25()