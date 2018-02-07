'''
tensorboard --logdir="./graphs/l2" --port 6006      see the tensorboard
http://localhost:6006/
'''
from utils import *
from model import *
from for_dict import *
import time
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

tf.flags.DEFINE_integer('time_step',300,'the sentence max length')
## train和test数据中train中句子最长的是285，test中是300，为了保持一致，这个地方需要设置成300
tf.flags.DEFINE_integer('embedding_size',100,'the word vector dimension')
tf.flags.DEFINE_integer('dictFeatureNum',5,'the dict feature number')
tf.flags.DEFINE_integer('dict_embedding_size', 50, 'the dict feature embedding size') ## 这里就相当于是一个5维到50维的映射
tf.flags.DEFINE_integer('hidden_size',50,'the single direction RNN output dimension')
tf.flags.DEFINE_integer('class_num', 9, 'the class number')
tf.flags.DEFINE_float('learning_rate',0.5,'the learning rate')
tf.flags.DEFINE_float('train_rate',0.8,'the percentage of all data')
tf.flags.DEFINE_string('w2v_model','gameAndyuliaokudata.model','the word angle w2v model')
tf.flags.DEFINE_integer('batch_size',50, 'the number of sentence fed to the network')
tf.flags.DEFINE_integer('epoch', 100, 'the number of using the all data')
tf.flags.DEFINE_float('keep_prob', 0.5, 'keep prob of dropout layer')
tf.flags.DEFINE_integer("checkpoint_every", 2, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 2, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_string('data_file','./data/train','the raw train data not contain the label')
tf.flags.DEFINE_string('label_file','./data/train_label','the label for train data')
tf.flags.DEFINE_string('vocab_dir','./data/vocab','the path to save the vocab dict json file')
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("is_train", True, 'this is a flag')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
max_len = FLAGS.time_step
docs, index_docs, labels, vocab = get_data(FLAGS.data_file, FLAGS.label_file, max_len)
# indexArray,labelArray,wordDict,rawSentences
word_vector = get_word_vector(vocab, FLAGS.w2v_model, FLAGS.embedding_size)
# array,第i个向量就是vocab中第i个词，mask和unk都被随机初始化了,word_vector这个其实就是提供一个lookup，包括unk和mask
jsObj = json.dumps(vocab)
if not os.path.exists(FLAGS.vocab_dir):
    os.makedirs(FLAGS.vocab_dir)
vocab_dir = os.path.join(FLAGS.vocab_dir, 'vocab')
fileObject = open(vocab_dir, 'w')
fileObject.write(jsObj)
fileObject.close()

FLAGS.vocab_num = len(vocab)
sentence_num, sentence_length = labels.shape
# FLAGS.time_step = sentence_length
FLAGS.time_step = 300

train_x = index_docs[: int(FLAGS.train_rate * sentence_num)]
train_y = labels[: int(FLAGS.train_rate * sentence_num)]
train_docs = docs[: int(FLAGS.train_rate * sentence_num)]
valid_x = index_docs[int(FLAGS.train_rate * sentence_num):]
valid_y = labels[int(FLAGS.train_rate * sentence_num):]
valid_docs = docs[int(FLAGS.train_rate * sentence_num):]

print("train_x shape:" + str(train_x.shape))
print("train_y shape:" + str(train_y.shape))
print("valid_x shape:" + str(valid_x.shape))
print("valid_y shape:" + str(valid_y.shape))
print(train_x[0])

##############################################使用上面处理的数据，下面训练模型##################################
def train_step(x, y, word_vector, dict_vector):
    feed_dict = {
        model.w2v_result: word_vector,
        model.dict_vector: dict_vector,
        model.x: x,
        model.y: y,
    }
    loss, step, _, loss_summary = sess.run([model.loss, model.global_step, model.train_op, model.summary_op], feed_dict)
    return loss, loss_summary

def validation_step(x, y, word_vector, valid_dict_vector):
    feed_dict = {
        model.w2v_result: word_vector,
        model.dict_vector:valid_dict_vector,
        model.x: x,
        model.y: y,
    }
    acc = sess.run(model.acc, feed_dict)
    # test_predict = tf.reshape(test_predict, [-1, FLAGS.time_step])
    ## 下面的操作会被画在graph里面，也就是lazy_loading的问题
    # acc = tf.reduce_mean(tf.cast(tf.equal(test_predict, y), dtype=tf.float32))
    # accRes = sess.run(acc)
    return acc

with tf.Graph().as_default():
    ## 配置GPU，是否打印日志等
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    # sess = tf.Session()
    with sess.as_default():                 ## 开始建立连续的会话机制
        model = AnnotationModel(FLAGS)      ## 读取Graph模型,只输出到了loss
        sess.run(tf.global_variables_initializer())
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp)) ## 模型保存路径:当前路径/runs/时间
        print('writing to {}\n'.format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        # checkpoint_prefix = os.path.abspath(os.path.join(checkpoint_dir, 'checkpoints'))
        # checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # if not os.path.exists(checkpoint_prefix):
        #     os.makedirs(checkpoint_prefix)
########################################变量初始化，准备加载图模型###################################################

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./graphs/l2', sess.graph)
        ret = batch_iter(train_x.shape[0], zip(train_x, train_y, train_docs), FLAGS.batch_size, FLAGS.epoch, False)
        ## 上面的ret是一个迭代器，是把epoch和batch都考虑到的一个结果
        saver = tf.train.Saver()
        file_list = ['trainSenDict']
        dictionary_list = get_dict(file_list)
        valid_dict_vector = matrix_index(valid_docs, dictionary_list, FLAGS.time_step)
        validationAccLog = {}
        patience_cnt = 0          ## 设置early_stopping防止过拟合
        trainLossPreStep = 0      ## 记录上一步的train loss，用来和当前步的loss进行比较，如果长时间不改变，就停止训练
        patience = 16
        for batch in ret:
            x_, y_, doc_ = zip(*batch[0])     ## zip(*) is equal to unzip
            dict_vector = matrix_index(doc_, dictionary_list, FLAGS.time_step)
            train_loss, loss_summary = train_step(x_, y_, word_vector, dict_vector)
            vali_acc = validation_step(valid_x,valid_y,word_vector, valid_dict_vector)
            current_step = tf.train.global_step(sess, model.global_step)
            ## 在tensorboard中画loss曲线
            writer.add_summary(loss_summary, global_step=current_step)
            epoch_step = current_step * FLAGS.batch_size / 20000
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                print('current_epoch is {}，current_step is {}， the train mean loss is {}'.format(epoch_step
        , current_step, train_loss))
                print('\nValidation Acc is:')
                print(vali_acc)
                validationAccLog[current_step] = vali_acc
########################################early stopping###################################################
                min_delta = 0.001
                if epoch_step > 0 and trainLossPreStep - train_loss > min_delta:
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                if patience_cnt > patience:
                    print("early stopping...")
                    break
                trainLossPreStep = train_loss
##########################################################################################################
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_dir, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
    writer.close()
    valiAccList = sorted(validationAccLog.items(), key=lambda d: d[1], reverse=True)
    accDir = os.path.abspath(os.path.join(checkpoint_dir, 'validationAccuracy'))
    with open(accDir,'w') as o:
        for item in valiAccList:
            o.write(str(item[0])+'\t'+str(item[1])+'\n')
    print("Optimization Finished!")
