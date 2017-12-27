# -*- coding: UTF-8 -*-
from __future__ import division
import tensorflow as tf
import argparse
import os
import cPickle as pickle
import numpy as np
import xlrd
import xlwt
from Dataloader import Dataloder
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--utils_dir', type=str, default='./utils',
                        help='''directory containing labels.pkl''')
    args = parser.parse_args()
    test(args)
def test(args):
    with open(os.path.join(args.utils_dir, 'config.pkl'), 'rb') as f:
        args = pickle.load(f)
    args.data_path = '../../Data/taiyi_test_sentiment.xlsx'
    args.sampling = True

    dataloader = Dataloder(isTraining=False, data_path=args.data_path, word2vec_dir=args.word2vec_path, batch_size=args.batch_size, num_steps=args.num_steps)
    model = Model(args)
    count_total = 0
    count_corrent = 0
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for row in dataloader.raw_data:
                sentence = row['sentence']
                themes = row['theme']
                try:
                    sentence_vector, _ = dataloader.create_one_data(row)
                except:
                    continue
                sentence_list = [k for _, k in enumerate(sentence)]
                result = model.predict_class(sess, sentence_vector)
                def getchar(a):
                    if  result[a] == 1:
                        return '[b]'+sentence_list[a]
                    elif result[a] == 2:
                        return '[m]' + sentence_list[a]
                    elif result[a] == 3:
                        return '[e]' + sentence_list[a]
                    elif result[a] == 4:
                        return '[s]' + sentence_list[a]
                    else:
                        return '*'
                out = [getchar(t) for t in range(min(len(sentence_list), len(result)))]
                out = ''.join(out)
                print sentence + '%' + out + '%' + themes

if __name__ == '__main__':
    main()