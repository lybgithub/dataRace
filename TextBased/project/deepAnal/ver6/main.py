# -*- coding: UTF-8 -*-
from __future__ import division
import time
import tensorflow as tf
import argparse
import cPickle as pickle
from Dataloader import Dataloder2
from Dataloader import writeList
from model import Model
from parse_file import method_2017_11_25
import csv
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 50, "batch_size")
flags.DEFINE_integer("num_steps", 250, "num_steps")
flags.DEFINE_integer("lstm_size", 256, "lstm_size")
flags.DEFINE_integer("num_layers", 1, "num_layers")
flags.DEFINE_integer("emb_size", 100, "emb_size")
flags.DEFINE_integer("grad_clip", 5, "grad_clip")
flags.DEFINE_integer("epochs", 80, "epochs")
flags.DEFINE_integer("save_every_n", 200, "save_every_n")
flags.DEFINE_integer("num_classes", 17, "num_classes")
flags.DEFINE_integer("vocab_size", 2471, "vocab_size")
flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
flags.DEFINE_float("min_learning_rate", 0.0005, "min_learning_rate")
flags.DEFINE_float("keep_prob", 0.5, "keep_prob")
flags.DEFINE_float("weight_decay", 0.005, "weight_decay")
flags.DEFINE_float('loss_weight', 5.0, "loss_weight")
flags.DEFINE_string("utils_dir", '../utils', "utils_dir")
flags.DEFINE_string("train_data_path", './Data/train_new_40000.csv', "train_data_path")
flags.DEFINE_string("train_data_label_path", './Data/train_new_40000_label.txt', "train_data_label_path")
flags.DEFINE_string("test_data_path", './Data/train_10000.csv', "test_data_path")
flags.DEFINE_string("test_data_label_path", './Data/train_10000_label.txt', "test_data_label_path")
flags.DEFINE_string("parse_data_path", './Data/taiyi_semi_test.csv', "parse_data_path")
flags.DEFINE_string("mode", '1', "1 for training, 2 for test, 3 for continue training, 4 for parse")
flags.DEFINE_string("save_dir", './save_1', "save_dir")
flags.DEFINE_string("word2vec_dir", '../word2vec', "word2vec_dir")
flags.DEFINE_bool("is_attention", True, "is_attention")
flags.DEFINE_bool("lr_decreased", False, "lr_decreased")
flags.DEFINE_bool("CRF", False, "has_CRF")
FLAGS = flags.FLAGS

def nostop():
    args = FLAGS
    args.save_dir = './save_1'
    train(args)
    args.batch_size = 20
    args.epochs = 60
    args.train_data_path = './Data/train_new_20000.csv'
    args.train_data_label_path = './Data/train_new_20000_label.csv'
    args.mode = '3'
    args.save_dir = './save_2'
    args.lr_decreased = True
    continue_training(args)

def main():
    args = FLAGS
    if args.mode == '1':
        train(args)
    elif args.mode == '2':
        test(args)
    elif args.mode == '3':
        continue_training(args)
    elif args.mode == '4':
        parse_test(args)
    else:
        pass

def continue_training(args):
    trained_cpkt_dir = './save_1'
    dataloader = Dataloder2(args=args)
    model = Model(args)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(trained_cpkt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            counter = 0
            for e in range(args.epochs):
                dataloader.reset_pointer()
                if args.lr_decreased:
                    args.learning_rate = args.learning_rate - dataloader.lr_decay
                for _ in range(dataloader.num_batch):
                    x, x_pre, x_pos, y, length, weight = dataloader.next_batch()
                    counter += 1
                    start = time.time()
                    feed = {model.inputs: x,
                            model.inputs_pre: x_pre,
                            model.inputs_post: x_pos,
                            model.targets: y,
                            model.inputs_length: length,
                            model.weights: weight,
                            model.learning_rate: args.learning_rate}
                    batch_loss, _ = sess.run([model.loss, model.optimizer], feed_dict=feed)
                    end = time.time()
                    if counter % 100 == 0:
                        print('round_num: {}/{}... '.format(e + 1, args.epochs),
                              'Training steps: {}... '.format(counter),
                              'Training error: {:.4f}... '.format(batch_loss),
                              'Learning rate: {:.4f}... '.format(args.learning_rate),
                              '{:.4f} sec/batch'.format((end - start)))
                    if (counter % args.save_every_n == 0):
                        saver.save(sess,
                                   "{path}/i{counter}_l{lstm_size}.ckpt".format(path=args.save_dir, counter=counter,
                                                                                lstm_size=args.lstm_size))
            saver.save(sess, "{path}/i{counter}_l{lstm_size}.ckpt".format(path=args.save_dir, counter=counter,
                                                                          lstm_size=args.lstm_size))

def parse_test(args):
    write = False
    args.save_dir = '/home/luofeng/PycharmProjects/DFCompetition/deepAnal/ver6/save_ver6_1209_newTrainData'
    # args.save_dir = './save_2017-11-26_17classes_100epoches_150numsteps_withAttention_2weight'
    dataloader = Dataloder2(args=args)
    model = Model(args)

    if write:
        save_file_path = './Data/res1203_ver6_5weight_continueTrainedOn20000_2.csv'
        head = ['row_id', 'content', 'theme', 'sentiment_word', 'sentiment_anls']
        csvFile = open(save_file_path, 'w')
        writer = csv.writer(csvFile)
        writer.writerow(head)
    result_list = []
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i in range(len(dataloader.raw_data)):
            # for i in range(2000):
                print i
                id = dataloader.raw_data[i]['id']
                content = dataloader.raw_data[i]['sentence'].decode('utf8')
                content_list = [c for _, c in enumerate(content)]
                x, x_pre, x_post, length = dataloader.create_one_data(dataloader.raw_data[i])
                predict = model.predict_class(sess, x, x_pre, x_post, length)
                result_list.append(predict[:len(content_list)].tolist())
                # themes, sentiments, tags = method_2017_11_25(content, content_list, predict)
                # print id, content, themes, sentiments, tags
                # if write:
                #     writer.writerow([str(id).encode('utf8'), str(content).encode('utf8'), str(themes).encode('utf8'),
                #                      str(sentiments).encode('utf8'), str(tags).encode('utf8')])
                #out = ''
                # for i in range(min(len(content), len(predict))):
                #    out = out + content_list[i] + '[' + str(predict[i]) + ']'
                #print content
                # print ' '.join([str(j) for j in y[0]])
                #print ' '.join([str(j) for j in predict])
                #print out
                #print '**********'
        if write:
            csvFile.close()
    writeList('./Data/predict_list_ver6.txt', result_list)
def test(args):
    args.save_dir = './save_2017-11-26_17classes_100epoches_150numsteps_withAttention'
    dataloader = Dataloder2(args=args)
    model = Model(args)
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i in range(dataloader.num_batch):
                content = dataloader.raw_data[i]['sentence']
                content_vector =[c for _, c in enumerate(content)]
                x, x_pre, x_post, y, length,weight = dataloader.next_batch()
                predict = model.predict_class(sess, x, x_pre, x_post, length)
                out1 = ''
                out2 = ''
                for i in range(min(len(content), len(predict))):
                    out1 = out1+content_vector[i] + '[' + str(y[0][i])+']'
                    out2 = out2+content_vector[i] + '[' + str(predict[i])+']'
                print content
                print ' '.join([str(j) for j in y[0]])
                print ' '.join([str(j) for j in predict])
                print 'y:' + out1
                print 'p' + out2
                print '**********'

def train(args):
    dataloader = Dataloder2(args=args)
    model = Model(args=args, embedding_matrix=dataloader.embedding_matrix)
    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        counter = 0
        for e in range(args.epochs):
            dataloader.reset_pointer()
            if args.lr_decreased:
                args.learning_rate = args.learning_rate - dataloader.lr_decay
            for _ in range(dataloader.num_batch):
                x, x_pre, x_post, y, length, weight = dataloader.next_batch()
                counter += 1
                start = time.time()
                feed = {model.inputs: x,
                        model.inputs_pre: x_pre,
                        model.inputs_post: x_post,
                        model.targets: y,
                        model.inputs_length: length,
			            model.weights: weight,
                        model.learning_rate: args.learning_rate}
                batch_loss, _ = sess.run([model.loss,model.optimizer],feed_dict=feed)

                end = time.time()
                # control the print lines
                if counter % 100 == 0:
                    print('round_num: {}/{}... '.format(e + 1, args.epochs),
                          'Training steps: {}... '.format(counter),
                          'Training error: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if (counter % args.save_every_n == 0):
                    saver.save(sess, "{path}/i{counter}_l{lstm_size}.ckpt".format(path = args.save_dir, counter=counter, lstm_size=args.lstm_size))
        saver.save(sess, "{path}/i{counter}_l{lstm_size}.ckpt".format(path = args.save_dir, counter=counter, lstm_size=args.lstm_size))
if __name__ == '__main__':
    main()
    # nostop()