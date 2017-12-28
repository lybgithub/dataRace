### 总体思路框架
&emsp;&emsp;首先我们把这个问题当做了一个序列标注sequence labeling的基础问题。采用的模型有两种：浅层模型(crf)+深层模型(word2vector+bilstm+3-gram+attention+jieba),考虑到在深层模型中加上crf之后，效果没有明显的提升，所以在比赛后期没有加上这一结构。 值得注意的是，对于深层模型的标注方式，我们采用的是BIESO的标注方式，从很大程度上提高了输出结果的精度。除此之外，我们使用训练出来的字向量，单独训练了一个情感分析模型，这个模型的准确率高达99%。</br>
&emsp;&emsp;在模型融合的时候，因为我们分析对于这个赛题，误判的代价会是漏判的两倍，所以我们使用了取交集的投票策略，事实证明，使用了这种策略之后，成绩得到了大幅提升。
### 代码运行环境
&emsp;&emsp;模型部分：python2.7，tensorflow1.2，规则部分：python3.5

#### 1、数据预处理，训练数据生成
&emsp;&emsp;运行/preprocessing/preprocess.py中的makeLabel()函数，函数中的data_path修改为训练数据路径，并修改相应保存文件路径。此步骤生成训练样本标签。
运行/preprocessing/preprocess.py中的makeSpeech()函数，函数中的data_path修改为训练数据路径，并修改相应保存文件路径。此步骤生成训练样本词性标签（最终parse的文件也要生成该文件）。
将上述生成好的文件置于/deepAnal/ver5/Data, /deepAnal/ver6/Data, /deepAnal/ver7/Data中
#### 2、字向量训练
&emsp;&emsp;使用gensim接口，skip-gram模式，embedding size设置为100，窗口的size设置为5，使用的corpus主要是
a.初赛以及复赛的训练测试数据
b.京东商品的评论数据
最后的模型相关文件在文件夹/deepAnal/word2vec
#### 3、训练阶段
&emsp;&emsp;我们训练了三个深度学习，ver5,ver6和ver7。思路是对评论中每个字进行分类。ver5只用了单个字进行训练，ver6用了3-gram进行训练，既每个字由其自身向量以及前后各一个字向量拼接组成，ver7在ver6的基础上加入了每个字的词性（由jieba分词获得）。 </br>
&emsp;&emsp;训练阶段，首先ver5(ver6、ver7)/main中的参数的batch_size设为50,num_steps设为150,lr_decreased设为False, train_data_path/train_data_label_path/train_data_speech_path设为前后期一起的40000条训练数据，mode设为1进行训练，得到初始模型;
然后batch_size设为10，num_steps设为150,lr_decreased设为True,</br> &emsp;&emsp;train_data_path/train_data_label_path/train_data_speech_path设为后期的20000条训练数据，mode设为3进行加训，得到最终模型;
然后num_steps设为250,mode设为4对最终提交数据进行parse，得到一版结果。

##### tips
&emsp;&emsp;训练阶段分别对ver5、ver6、ver7设置不同的训练参数，可以得到不同版本的结果(该结果中只含有每个字的类别)。
#### 4、模型融合
&emsp;&emsp;在第3步中得到的结果放入ver6/Data/fortoupiao中，运行/ver6/toupiao.py，得到结果文件（该结果是可用于提交的文件格式）
tips:由于训练时间较久，不利于复现。前期模型的结果已经保存在/ver6/Data/fortoupiao文件夹中。复现时可修改ver6/toupiao.py中main()函数里的dir路径，直接得到结果文件。

#### 5、异常值处理
&emsp;&emsp;对于一些主题情感结果来说，模型不能提取出来，比如“没有想象中的好”，所以设定了一些规则进行处理，不过这部分处理在分数的提升上只是很小很小，也就一个千分点，
主要的得分还是依靠模型，这部分文件在/rule中，其中rules.csv是规则项，在manual_rules.py中，直接输入模型处理出的结果，最后输出的out文件就是最终的结果.

#### 6、亮点
##### 第一
&emsp;&emsp;使用attention机制应用到sequence model上面，从一定程度上小幅提升准确率。
##### 第二
&emsp;&emsp;使用3-gram的策略，更深一步应用上下文信息，较大幅度提高准确率。
##### 第三
&emsp;&emsp;对训练数据进行筛选，去除可能会对模型产生干扰的数据，主要是因为初赛的数据质量不太好。除此之外，我们还使用了“过采样”的思路解决样本数据不够的问题，也就是使用情感词典中的低频情感词，替换数据中的高频情感词作为新的训练样本，事实证明这种做法有提高。
##### 第四
&emsp;&emsp;把情感倾向和情感词一并训练，具体思路就是：每一个情感词的label不仅仅标注情感词，还会标注倾向。比如：喜欢这个情感词，在不考虑情感倾向的情况下，这个词的标注是1 2,1代表情感词的开始，2代表情感词的结束。那么一同训练的意思就是说，情感词的开始会继续对应三个维度，正向情感词的开始，中性情感词的开始以及负性情感词的开始。
