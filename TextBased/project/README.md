### 代码运行环境
模型部分：python2.7，tensorflow1.2，规则部分：python3.5

#### 1、数据预处理，训练数据生成
运行/preprocessing/preprocess.py中的makeLabel()函数，函数中的data_path修改为训练数据路径，并修改相应保存文件路径。此步骤生成训练样本标签。
运行/preprocessing/preprocess.py中的makeSpeech()函数，函数中的data_path修改为训练数据路径，并修改相应保存文件路径。此步骤生成训练样本词性标签（最终parse的文件也要生成该文件）。
将上述生成好的文件置于/deepAnal/ver5/Data, /deepAnal/ver6/Data, /deepAnal/ver7/Data中
#### 2、字向量训练
使用gensim接口，skip-gram模式，embedding size设置为100，窗口的size设置为5，使用的corpus主要是
a.初赛以及复赛的训练测试数据
b.京东商品的评论数据
最后的模型相关文件在文件夹/deepAnal/word2vec
#### 3、训练阶段
我们训练了三个深度学习，ver5,ver6和ver7。思路是对评论中每个字进行分类。ver5只用了单个字进行训练，ver6用了3-gram进行训练，既每个字由其自身向量以及前后各一个字向量拼接组成，ver7在ver6的基础上加入了每个字的词性（由jieba分词获得）。
训练阶段，首先ver5(ver6、ver7)/main中的参数的batch_size设为50,num_steps设为150,lr_decreased设为False, train_data_path/train_data_label_path/train_data_speech_path设为前后期一起的40000条训练数据，mode设为1进行训练，得到初始模型;
然后batch_size设为10，num_steps设为150,lr_decreased设为True, train_data_path/train_data_label_path/train_data_speech_path设为后期的20000条训练数据，mode设为3进行加训，得到最终模型;
然后num_steps设为250,mode设为4对最终提交数据进行parse，得到一版结果。

##### tips
训练阶段分别对ver5、ver6、ver7设置不同的训练参数，可以得到不同版本的结果(该结果中只含有每个字的类别)。
#### 4、模型融合
在第3步中得到的结果放入ver6/Data/fortoupiao中，运行/ver6/toupiao.py，得到结果文件（该结果是可用于提交的文件格式）
tips:由于训练时间较久，不利于复现。前期模型的结果已经保存在/ver6/Data/fortoupiao文件夹中。复现时可修改ver6/toupiao.py中main()函数里的dir路径，直接得到结果文件。

#### 5、异常值处理
对于一些主题情感结果来说，模型不能提取出来，比如“没有想象中的好”，所以设定了一些规则进行处理，不过这部分处理在分数的提升上只是很小很小，也就一个千分点，
主要的得分还是依靠模型，这部分文件在/rule中，其中rules.csv是规则项，在manual_rules.py中，直接输入模型处理出的结果，最后输出的out文件就是最终的结果.
