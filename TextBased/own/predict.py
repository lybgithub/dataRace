from utils import *
from model import *
from for_dict import *
import json
import os
import codecs
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

tf.flags.DEFINE_string('predict_result', './result/res1115', 'the predict result')
tf.flags.DEFINE_string('test_file','./data/test','the file to predict')
tf.flags.DEFINE_string('vocab_dir','./data/vocab/vocab','the path to save the vocab dict json file')
tf.flags.DEFINE_string('train_label_file', './data/train_label', 'the label for train data')
tf.flags.DEFINE_string("model_name", "./runs/1510717576/checkpoints-4600", "Name(Path) of model")
tf.flags.DEFINE_string('w2v_model','gameAndyuliaokudata.model','the word angle w2v model')
tf.flags.DEFINE_integer('embedding_size',100,'the word vector dimension')
tf.flags.DEFINE_integer('time_step',300,'the max sequence length of the train and test data')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

with open(FLAGS.vocab_dir, "r") as f:
    vocab = json.load(f)

with open(FLAGS.train_label_file,'r') as f:
    labels = f.readlines()
with codecs.open(FLAGS.test_file, 'r', 'utf8') as f:
    test = f.readlines()
max_len = FLAGS.time_step
index_test = get_test_data(FLAGS.test_file, vocab, max_len)
file_list = ['trainSenDict']
dictionary_list = get_dict(file_list)
test_dict_series = matrix_index(test, dictionary_list, FLAGS.time_step)
# indexArray,labelArray
word_vector = get_word_vector(vocab, FLAGS.w2v_model, FLAGS.embedding_size)
# array,第i个向量就是vocab中第i个词，mask和unk都被随机初始化了
###########################处理好需要预测的数据，准备预测##########################################
g1 = tf.Graph()
sess1 = tf.Session(graph=g1)
with sess1.as_default():
    with g1.as_default():
        print("begin import the meta graph")
        saver_main = tf.train.import_meta_graph(FLAGS.model_name + ".meta")
        print("begin restoring the train model")
        saver_main.restore(sess1, FLAGS.model_name)

        feed_dict = {
            g1.get_tensor_by_name('w2v_result:0'): word_vector,
            g1.get_tensor_by_name('input:0'): index_test,
            g1.get_tensor_by_name('dictSeries:0'): test_dict_series,
        }
        result = g1.get_tensor_by_name('properties_softmax_loss/softmax_result2:0')
        test_predict = sess1.run(result, feed_dict)
###################################get the end result########################################
        arr2 = test_predict.reshape((20000, 300))
        print(arr2[114])
        o = codecs.open(FLAGS.predict_result, 'w', 'utf8')
        test_data = codecs.open(FLAGS.test_file, 'r', 'utf8')
        lines = test_data.readlines()
        sentiment = []
        theme = []
        for i, line in enumerate(lines):
            sentence_len = len(line.strip())
            res = arr2[i][:sentence_len+1]  ##方便后面的提取处理
            sen_left = 0
            the_left = 0
            sen_flag = 0
            the_flag = 0
            # print(res)
            for j, tag in enumerate(res):
                if(j==len(res)-1):
                    break
                if(tag==1 and sen_flag==0):
                    sen_left = j
                    sen_flag = 1
                    if(res[j+1]!=2 and res[j+1]!=3):
                        sen_flag = 0
                elif(tag==2 and sen_flag==1):
                    if(res[j+1]!=2 and res[j+1]!=3):
                        sen_flag = 0
                elif(tag==3 and sen_flag==1):
                    sentiment.append(line[sen_left:j+1])
                    sen_flag = 0
                elif(tag==4):
                    sentiment.append(line[j])
                elif(tag==5 and the_flag==0):
                    the_left = j
                    the_flag = 1
                    if (res[j+1]!=6 and res[j+1]!=7):
                        the_flag = 0
                elif(tag==6 and the_flag==1):
                    if (res[j+1]!=6 and res[j+1]!=7):
                        the_flag = 0
                elif(tag==7 and the_flag==1):
                    theme.append(line[the_left:j+1])
                    the_flag = 0
                elif(tag==8):
                    theme.append(line[j])
            if(len(theme)==0):
                o.write('NULL' + '\t' + '\t' + ';'.join(sentiment) + '\n')
            else:
                o.write(';'.join(theme) + '\t'+'\t' + ';'.join(sentiment) + '\n')
            sentiment = []
            theme = []
        o.close()
print('predict is ok!')
# if (tag == 1 and sen_flag == 0):
#     sen_left = j
#     sen_flag = 1
# if (tag != 1 and sen_flag == 1):
#     sentiment.append(line[sen_left:j+1])
#     sen_flag = 0
# if (tag == 2 and the_flag == 0):
#     the_left = j
#     the_flag = 1
# if (tag != 2 and the_flag == 1):
#     theme.append(line[the_left:j+1])
#     the_flag = 0