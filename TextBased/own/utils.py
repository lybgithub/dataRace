#coding=utf-8
import codecs
import numpy as np
import gensim
import random
from collections import OrderedDict

#######################process the raw data file#############################
def get_data(data_file,label_file,max_len):
    maskid = 0    ## padding class
    f = codecs.open(data_file,'r','utf8')
    f2 = codecs.open(label_file, 'r', 'utf8')
    docs = f.readlines()   ## list
    vocab = gen_vocab(docs, maskid=0)  ## dict
    index_docs = word2index(docs, vocab, max_len)  ## list

    labels = f2.readlines()
    # max_len = max([len(label) for label in labels])
    labelRes = []
    for label in labels:
        label = list(label.strip())
        label = [int(tag) for tag in label]
        label = label + [maskid] * (max_len - len(label))
        labelRes.append(label)
    # labels = [list(int(tag) for tag in label.strip()) + [maskid] * (max_len - len(label)) for label in labels]
    labels = np.array(labelRes)
    ## list [[00111222222],[1110000002222]]
    return np.array(docs),np.array(index_docs),labels,vocab

def get_test_data(test_file,vocab,train_max_len):
    f = codecs.open(test_file, 'r', 'utf8')
    docs = f.readlines()   ## list
    index_docs = word2index(docs, vocab, train_max_len)  ## list
    res = np.array(index_docs)
    return res

## 可以考虑过滤低词频的字
def gen_vocab(docs, maskid=0):
    word_set = set(word for doc in docs for word in doc.strip())  # ['中' ‘美’ ‘国’]
    vocab = {word: index + 1 for index, word in enumerate(word_set)}
    vocab['mask'] = maskid  # add 'mask' padding word
    vocab[u'unk'] = len(vocab)  # add 'unk'
    return vocab    ## dict {'unk':2873,'mask':0}

def word2index(docs, vocab, train_max_len=0):
    index_docs = []
    for doc in docs:
        index_list = []
        for word in doc:
                if word not in vocab:
                        word = u'unk'
                index_list.append(vocab[word])
        index_docs.append(index_list)
    if(train_max_len==0):                                   ## 对于测试集的时候直接传入与训练集一致的padding长度~
        train_max_len = max([len(doc) for doc in index_docs])
    index_docs = [doc + [vocab['mask']] * (train_max_len - len(doc)) for doc in index_docs]
    return index_docs   ## list [[2380,  512, 1254, ...,    0,    0,    0],[1548,...]]

#########################w2v_model#######################################
def get_word_vector(vocab, w2v_model, embedding_size):
    word_vector = []
    unkown_vector = [i/(embedding_size+0.0) for i in range(embedding_size)]
    word_vector.append(unkown_vector)
    wv_model = gensim.models.Word2Vec.load(w2v_model)
    #sorted by value
    vocab_ordered = OrderedDict(sorted(vocab.items(), key=lambda x: x[1]))  # 按照索引大小进行排序
    for word in vocab_ordered.keys():
        if word in wv_model:
                word_vector.append(wv_model[word])
        else:
                word_vector.append([random.random() for i in range(embedding_size)])
    word_vector.pop()
    return np.array(word_vector)   ## 按照索引的向量字典，字典里面没有的直接随机初始化

#######################################生成一个batch_size的迭代器##########################################
def batch_iter(data_size, data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(list(data))
    #data_size = len(data)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)   # 解决最后一个batch溢出的问题
            yield (shuffled_data[start_index:end_index], start_index)

