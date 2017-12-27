# -*- coding: utf-8 -*-
import os, sys
import time
import heapq
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib import rnn
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'

class Model():
    def __init__(self,  args, embedding_matrix = None):
        self.mode = args.mode
        self.is_attention = args.is_attention
        self.CRF = args.CRF
        self.num_steps = args.num_steps
        self.num_layers = args.num_layers
        self.lstm_size = args.lstm_size
        self.num_classes = args.num_classes
        self.grad_clip = args.grad_clip
        self.vocab_size = args.vocab_size
        self.speech_size = args.speech_size
        self.emb_size = args.emb_size
        self.speech_emb_size = args.speech_emb_size
        self.weight_decay = args.weight_decay
        if self.mode == '1' or self.mode == '3':
            self.batch_size = args.batch_size
            self.keep_prob = args.keep_prob
        else:
            self.batch_size = 1
            self.keep_prob = 1.0
        self.build_variable(embedding_matrix)
        self.build_model()

    def build_variable(self,embedding_matrix):
        self.cell_fw, self.cell_bw = self.build_lstm()
        self.inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='input')
        self.inputs_pre = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='input_pre')
        self.inputs_post = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='input_post')
        self.inputs_speech = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='input_speech')
        self.targets = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='targets')
        self.inputs_length = tf.placeholder(tf.int32, shape=(self.batch_size), name='input_length')
        self.weights = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_steps), name='weight')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        if self.mode == '1':
            self.embedding = tf.Variable(embedding_matrix, trainable=True, name="emb", dtype=tf.float32)
        else:
            self.embedding = tf.get_variable("emb", [self.vocab_size, self.emb_size])
        self.speech_embedding = tf.get_variable("speech_emb", [self.speech_size, self.speech_emb_size])
        if self.is_attention:
            with tf.variable_scope('softmax'):
                self.softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size * 4, self.num_classes], stddev=0.1),name='weight') #从截断的正态分布中输出随机值。 生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
                self.softmax_b = tf.Variable(tf.zeros(self.num_classes), name='bias')
        else:
            with tf.variable_scope('softmax'):
                self.softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size * 2, self.num_classes], stddev=0.1),name='weight') #从截断的正态分布中输出随机值。 生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
                self.softmax_b = tf.Variable(tf.zeros(self.num_classes), name='bias')
        if self.CRF:
            with tf.variable_scope('CRF'):
                self.transition_params = tf.get_variable("transition_params", [self.num_classes, self.num_classes])
    def build_model(self):
        inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.inputs)
        inputs_pre_embedded = tf.nn.embedding_lookup(self.embedding, self.inputs_pre)
        inputs_post_embedded = tf.nn.embedding_lookup(self.embedding, self.inputs_post)
        inputs_speech_embedded = tf.nn.embedding_lookup(self.speech_embedding, self.inputs_speech)
        inputs = tf.concat([inputs_embedded, inputs_pre_embedded, inputs_post_embedded, inputs_speech_embedded], axis=-1)
        outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
            self.cell_fw,
            self.cell_bw,
            inputs,
            sequence_length=self.inputs_length,
            dtype=tf.float32)
        self.outputs = tf.concat(outputs, 2)
        if self.is_attention:
            self.outputs = self.attention_output(self.outputs)
        self.logits = self.build_output(self.outputs)
        self.prediction = tf.nn.softmax(self.logits, name='probabilites')
        self.loss = self.build_loss(self.logits, self.targets)
        self.optimizer = self.build_optimizer(self.loss)

    def attention_output(self, outputs):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(2*self.lstm_size, outputs, memory_sequence_length=self.inputs_length)
        outputs_split = tf.split(outputs, self.num_steps, axis=1)
        context = [tf.squeeze(tf.matmul(tf.expand_dims(attention_mechanism(tf.squeeze(outputs_split[i], axis=1), None), 1),outputs),1) for i in range(self.num_steps)]
        context = tf.stack(context,1)
        outputs_attentioned = tf.concat([outputs, context],axis=2)
        return outputs_attentioned

    def build_lstm(self):

        lstm_size = self.lstm_size
        num_layer = self.num_layers
        keep_prob = self.keep_prob

        # lstm cell
        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=0.5)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=0.5)

        # dropout
        lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(keep_prob))
        lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(keep_prob))

        lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * num_layer)
        lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * num_layer)

        return lstm_cell_fw, lstm_cell_bw
    def inputs_reshape(self, inputs):
        lstm_size = self.lstm_size
        num_seqs = self.batch_size
        num_steps = self.num_steps
        feature_dim = self.feature_dim
        input_1 = tf.reshape(inputs, [-1, feature_dim])
        with tf.variable_scope('reshape_layer'):
            softmax_w = tf.Variable(tf.truncated_normal([feature_dim, lstm_size], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(lstm_size))
        input_2 = tf.matmul(input_1, softmax_w) + softmax_b
        input_3 = tf.reshape(input_2, [num_seqs, num_steps, lstm_size])
        return input_3
    def build_output(self, output):
        batch_size = self.batch_size
        num_steps = self.num_steps
        lstm_size = self.lstm_size  #in_size
        num_classes = self.num_classes  #out_size
        if self.is_attention:
            x = tf.reshape(output, [-1, lstm_size*4])
        else:
            x = tf.reshape(output, [-1, lstm_size * 2])
        # with tf.variable_scope('softmax'):
        #     softmax_w = tf.Variable(tf.truncated_normal([lstm_size * 2, num_classes], stddev=0.1),name='weight') #从截断的正态分布中输出随机值。 生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
        #     softmax_b = tf.Variable(tf.zeros(num_classes), name='bias')
        # 计算logits
        logits = tf.matmul(x, self.softmax_w) + self.softmax_b
        logits = tf.reshape(logits, [batch_size, num_steps, num_classes])
        return logits
    def build_loss(self, logits, targets):

        # weights = tf.sequence_mask(self.inputs_length, self.num_steps, dtype=tf.float32)
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=logits,
                                                                        tag_indices=targets,
                                                                        sequence_lengths=self.inputs_length,
                                                                        transition_params=self.transition_params)
            loss = -tf.reduce_mean(log_likelihood)
        else:
            loss = tf.contrib.seq2seq.sequence_loss(logits, targets, self.weights)
        return loss
    def build_optimizer(self, loss):
        learning_rate = self.learning_rate
        grad_clip = self.grad_clip

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))

        return optimizer

    def getL2Reg(self):
        l2 = sum(
            tf.contrib.layers.l2_regularizer(self.weight_decay)(tf_var)
            for tf_var in tf.trainable_variables()
            if not ("noreg" in tf_var.name or "bias" in tf_var.name)
        )
        return l2

    def predict_class(self, sess, sentence_vector, sentence_vector_pre, sentence_vector_post, speech, sentence_length):

        feed = {self.inputs: sentence_vector,
                self.inputs_length: sentence_length,
                self.inputs_pre: sentence_vector_pre,
                self.inputs_post: sentence_vector_post,
                self.inputs_speech: speech}
        if not self.CRF:
            probs= sess.run([self.prediction], feed_dict=feed)
            probs = np.squeeze(probs)
            probability = np.max(probs, axis=1)
            results = np.argmax(probs, 1)
            return results, probability
        else:
            logits, transition_param = sess.run([self.logits, self.transition_params], feed_dict=feed)
            viterbi_seq, _ = viterbi_decode(logits[0][:sentence_length[0]], transition_param)
            return np.array(viterbi_seq), None
    def batch_predict_class(self, sess, sentence_vector, sentence_vector_pre, sentence_vector_post, speech, sentence_length):

        feed = {self.inputs: sentence_vector,
                self.inputs_length: sentence_length,
                self.inputs_pre: sentence_vector_pre,
                self.inputs_post: sentence_vector_post,
                self.inputs_speech: speech}
        probs= sess.run([self.prediction], feed_dict=feed)
        probs = np.squeeze(probs)
        results = np.argmax(probs, -1)
        return results
