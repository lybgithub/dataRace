from utils import *
from model import *
from for_dict import *
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

tf.flags.DEFINE_integer('embedding_size',100,'the word vector dimension')
tf.flags.DEFINE_integer('dictFeatureNum',5,'the dict feature number')
tf.flags.DEFINE_integer('class_num', 9, 'the class number')
tf.flags.DEFINE_integer('dict_embedding_size', 50, 'the dict feature embedding size')
tf.flags.DEFINE_string('fineTuneData', './data/train', 'the fine tune data')
tf.flags.DEFINE_string('fineTuneLabel', './data/train_label', 'the fine tune label of the data above')
tf.flags.DEFINE_string('vocab_dir','./data/vocab/vocab','the path to save the vocab dict json file')
tf.flags.DEFINE_string("model_name", "./runs/1512141342/checkpoints-414", "Name(Path) of model")
tf.flags.DEFINE_string('w2v_model','gameAndyuliaokudata.model','the word angle w2v model')
tf.flags.DEFINE_integer('time_step',300,'the max sequence length of the train and test data')
tf.flags.DEFINE_integer('batch_size',50, 'the number of sentence fed to the network')
tf.flags.DEFINE_float('learning_rate',0.5,'the learning rate')
tf.flags.DEFINE_float('train_rate',0.8,'the percentage of all data')
tf.flags.DEFINE_integer('epoch', 500, 'the number of using the all data')
tf.flags.DEFINE_float('keep_prob', 0.5, 'keep prob of dropout layer')
tf.flags.DEFINE_integer("checkpoint_every", 2, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 2, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer('hidden_size',50,'the single direction RNN output dimension')


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

max_len = FLAGS.time_step

docs, index_docs, labels, vocab = get_data(FLAGS.fineTuneData, FLAGS.fineTuneLabel, max_len)

with open(FLAGS.vocab_dir, "r") as f:
    vocab = json.load(f)
FLAGS.vocab_num = len(vocab)
file_list = ['trainSenDict']
dictionary_list = get_dict(file_list)
fineTune_dict_series = matrix_index(docs, dictionary_list, FLAGS.time_step)
# indexArray,labelArray
word_vector = get_word_vector(vocab, FLAGS.w2v_model, FLAGS.embedding_size)
# array,第i个向量就是vocab中第i个词，mask和unk都被随机初始化了
###########################处理好需要预测的数据，准备预测##########################################
sess=tf.Session()
#First let's load meta graph and restore weights
print("begin import the meta graph")
saver = tf.train.import_meta_graph(FLAGS.model_name + ".meta")
graph = tf.get_default_graph()
with graph.as_default():
    with sess.as_default():
        print("begin restoring the train model")
        saver.restore(sess, FLAGS.model_name)
        model = AnnotationModel(FLAGS)  ## 读取Graph模型,只输出到了loss
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))  ## 模型保存路径:当前路径/runs/时间
        print('writing to {}\n'.format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'fineTuneCheckpoints'))
        sess.run(tf.global_variables_initializer())
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
#######################################restore graph is ok!##########################
            ret = batch_iter(docs.shape[0], zip(index_docs, labels, docs), FLAGS.batch_size, FLAGS.epoch, False)
            saver = tf.train.Saver()
            patience_cnt = 0  ## 设置early_stopping防止过拟合
            trainLossPreStep = 0  ## 记录上一步的train loss，用来和当前步的loss进行比较，如果长时间不改变，就停止训练
            patience = 16
            print('go fine tune')
            for batch in ret:
                x_, y_, doc_ = zip(*batch[0])  ## zip(*) is equal to unzip
                dict_vector = matrix_index(doc_, dictionary_list, FLAGS.time_step)
                feed_dict = {
                    model.w2v_result: word_vector,
                    model.dict_vector: dict_vector,
                    model.x: x_,
                    model.y: y_,
                }
                loss, step, _ = sess.run([model.loss, model.global_step, model.train_op],feed_dict)
                epoch_step = step * FLAGS.batch_size / 20000
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    print('current_epoch is {}，current_step is {}， the train mean loss is {}'.format(epoch_step, step,loss))
    ##########################################early stopping############################################################################################
                min_delta = 0.001
                if epoch_step > 0 and trainLossPreStep - loss > min_delta:
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                if patience_cnt > patience:
                    print("early stopping...")
                    break
                trainLossPreStep = loss
    ##################################################################################################################################################
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_dir, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))
            print("Optimization Finished!")



