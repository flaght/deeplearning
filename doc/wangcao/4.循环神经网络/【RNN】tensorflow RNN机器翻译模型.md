# 【RNN】tensorflow RNN机器翻译模型

标签（空格分隔）： 王小草深度学习笔记

---

时间：2016年10月27日
笔记整理者：王小草

我想，周四，对于上班族和上学族而言，应是一个很暧昧的日子，像极了初春的诱惑，蠢动又趋于安静，切切不安得期待周五的来临。确切的说，是周五的晚上的来临。那可以是多么狂热与放纵的一晚。内心可以是特别自由，轻松，为所欲为，哪怕熬夜，喝酒，或是通宵学习，都觉得没那么累。这么说来，其实我们大部分时候的疲惫，都来自于内心的魔鬼。

这一周我都在整理RNN相关的笔记。对于公司给我们自己学习的时间及其珍惜与感恩。回到家9点吃好晚饭10点洗好碗洗好澡11点来不及45度忧伤就要倒头入睡等待明天6点的闹钟不开心地把我叫起来，然后飞奔到地铁起点站开始我两个半小时的轨道迷途。这是我的日常，精神好的时候能够看看书看看代码，要是睡意还没又褪去，我就会像周遭人那样坐着进入梦乡，才不管什么形象。

我明白大城市的奔波，也明白大城市的公平。

---

##基本介绍
循环神经网络的笔记已经有好多篇，循序渐进的依次阅读的顺序是：
《王小草【深度学习】笔记第六弹-循环神经网络RNN与LSTM》
《王小草【深度学习】笔记第七弹-循环是网络的应用：注意力模型与机器翻译》
《【RNN】神奇的循环时间网络RNN-Python代码解析》
《【RNN】tensorflow实现循环神经网络RNN-PTB案例代码解析》
《【RNN】tensorflow RNN 机器翻译模型》（本文）

许多地方都是受教于寒小阳老师，在此表示万分感恩。

本文讲的是RNN实现机器翻译的tensorflow 代码。相对应的RNN 翻译系统的理论可以在《王小草【深度学习】笔记第七弹-循环神经网络的应用：注意力模型与机器翻译》的第二章中找到。说起机器翻译，我们的日常都不陌生，也许每天都会用到有道，百度等翻译工具，对于专业的翻译工作者，如果计算机能帮助他们实现精准地翻译，将极大提高工作效率。

本文的官方的教学文档地址：https://www.tensorflow.org/versions/r0.11/tutorials/seq2seq/index.html
官方github上的代码地址：https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/rnn/translate

本文的案例中使用了WMT'15 Website的数据来训练，这个数据下载下来有20G大小，所以，这个模型，慎跑啊！！

这个模型总共涉及以下这些代码（代码的位置上面链接中给出了）：
seq2seq_model.py --建立序列到序列的神经网络翻译模型
data_util--预处理与准备训练的数据
translate.py--调用以上方法运行翻译模型

接下去，会对每份代码进行讲解。如果计算机性能好的话，或者有GPU的话，可以尝试运行一下。

## 1.data_util.py
这个包主要是做数据的预处理，许多地方值得我们学习，过程大致如下：

下载数据-->解压数据-->建立语料库词典-->将训练集与测试集中的word都转换成id --保存数据，并返回路径

###1.1 导入需要的包
```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf

```
###1.2准备工作
定义一些特殊的字符串
定义两个正则化的表达式
下载数据的链接
```
# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

# URLs for WMT data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"

```

###2.3下载与解压数据
下载数据，如果已经存在就不重复下载了。
```
def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not os.path.exists(directory):
    print("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not os.path.exists(filepath):
    print("Downloading %s to %s" % (url, filepath))
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Succesfully downloaded", filename, statinfo.st_size, "bytes")
return filepath
```
解压数据到新的路径下
```
def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  print("Unpacking %s to %s" % (gz_path, new_path))
  with gzip.open(gz_path, "rb") as gz_file:
    with open(new_path, "wb") as new_file:
      for line in gz_file:
new_file.write(line)
```
调用以上两个方法，将训练集下载下来并且解压，总共有两个文件，一个英文，一个是法文。
返回测试集存放的路劲
```
def get_wmt_enfr_train_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  train_path = os.path.join(directory, "giga-fren.release2.fixed")
  if not (gfile.Exists(train_path +".fr") and gfile.Exists(train_path +".en")):
    corpus_file = maybe_download(directory, "training-giga-fren.tar",
                                 _WMT_ENFR_TRAIN_URL)
    print("Extracting tar file %s" % corpus_file)
    with tarfile.open(corpus_file, "r") as corpus_tar:
      corpus_tar.extractall(directory)
    gunzip_file(train_path + ".fr.gz", train_path + ".fr")
    gunzip_file(train_path + ".en.gz", train_path + ".en")
return train_path
```
下载并解压测试集，返回数据集存放的路劲
```
def get_wmt_enfr_dev_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  dev_name = "newstest2013"
  dev_path = os.path.join(directory, dev_name)
  if not (gfile.Exists(dev_path + ".fr") and gfile.Exists(dev_path + ".en")):
    dev_file = maybe_download(directory, "dev-v2.tgz", _WMT_ENFR_DEV_URL)
    print("Extracting tgz file %s" % dev_file)
    with tarfile.open(dev_file, "r:gz") as dev_tar:
      fr_dev_file = dev_tar.getmember("dev/" + dev_name + ".fr")
      en_dev_file = dev_tar.getmember("dev/" + dev_name + ".en")
      fr_dev_file.name = dev_name + ".fr"  # Extract without "dev/" prefix.
      en_dev_file.name = dev_name + ".en"
      dev_tar.extract(fr_dev_file, directory)
      dev_tar.extract(en_dev_file, directory)
return dev_path
```
###2.4 建立词典
**对一句话中的词进行分词**
```
def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
return [w for w in words if w]
```
**建立词典**（如果不存在的话）
原始的数据文件中的格式应是每行一句话，每句话会被分词，然后每个词可以初始化为数值0（可选的）。这个词典中最后会根据设置的max_vocabulary_size，依照词频大小包含前n个最高词频的词。然后这个词典会被保存在给定的路径中，格式是每一行一个词。所以第一个词的索引是0，第二个词的索引是1，以此类推。

输入的参数有：
vocabulary_path：词典将要存放的路径
data_path:原始数据存放的路径
max_vocabulary_size：词典的大小
tokenizer：分词的方法，如果为none的话，就会使用默认的方法basic_tokenizer
normalize_digits：布尔类型，如果为true的话，那么所有数字会替代为0
```
def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  # 如果不存在词典的路径
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    # 建立一个空的字典，用来存放word
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      # 对于原数据中的每一行（也就是每一句话）
      for line in f:
        # 数一下有多少句话，没100000句的时候打印以下，以方便我们跟踪
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        # 对每句话分词
        line = tf.compat.as_bytes(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        # 对这句话中的所有数字都转换为0，如果normalize_digits=True，否则就保持原样
        for w in tokens:
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          # 如果这个词在字典vocab中有了，那么就加一个计数
          if word in vocab:
            vocab[word] += 1
          # 如果没有的话就增加这个词为key,value为1
          else:
            vocab[word] = 1
      # 降序排序
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      # 如果词列表大于字典预设的最大值，那么就取出前n个词
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      # 保存到词典的路径中，每个词一行
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")
```
**建立索引**
读取每一行的词，然后维每个词带上索引

输入的参数有：
vocabulary_path：词典的路径

返回的是：
（词，索引）
（索引，词）

```
def initialize_vocabulary(vocabulary_path):
 
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else: # 如果词典不存在就报错吧
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

```

###2.5 将文本中的词根据字典转换成对应的索引
比如将一句话["I", "have", "a", "dog"] ，根据字典中的索引{"I": 1, "have": 2,
  "a": 4, "dog": 7"}，转换成[1, 2, 4, 7]
  
输入的参数有：
sentence：一句话（bytes format）
vocabulary：带有索引的词典（词，索引）
tokenizer：分词的方法，如果为none的话，就会使用默认的方法basic_tokenizer
normalize_digits：布尔类型，如果为true的话，那么所有数字会替代为0

返回的是：
转换成id后的词列表
```
def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):

  # 分词
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  # 标准化数字（可选），然后将所有词转换成对应的id
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]

```
以下方法是将原始数据一行一行读入，然后调用上面sentence_to_token_ids方法和词典将句子中的词都转换成id,然后把结果保存到target_path中。

输入的参数：
data_path:下载下来的原始数据保存到路径
target_path:转换成id的数据要保存的路径
vocabulary：带有索引的词典（词，索引）
tokenizer：分词的方法，如果为none的话，就会使用默认的方法basic_tokenizer
normalize_digits：布尔类型，如果为true的话，那么所有数字会替代为0

```
def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
 
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

```
###2.6 调用以上所有方法，对原始数据进行预处理

输入的参数有：
data_dir: 所有数据集存储的路径
en_vocabulary_size: 英文字典的大小
fr_vocabulary_size: 法文字典的大小
tokenizer：分词的方法，如果为none的话，就会使用默认的方法basic_tokenizer

返回的是：
6个元素的tuple
      (1) 以token-ids形式存储的英文训练集的路径
      (2) 以token-ids形式存储的法文训练集的路径
      (3) 以token-ids形式存储的英文测试集的路径
      (4) 以token-ids形式存储的法文测试集的路径
      (5) 英文字典的路径
      (6) 法文字典的路径

```
def prepare_wmt_data(data_dir, en_vocabulary_size, fr_vocabulary_size, tokenizer=None):
  
  # 根据路径获取训练集与测试集原始数据
  train_path = get_wmt_enfr_train_set(data_dir)
  dev_path = get_wmt_enfr_dev_set(data_dir)

  # 分别根据训练集创建英文字典与法文字典
  fr_vocab_path = os.path.join(data_dir, "vocab%d.fr" % fr_vocabulary_size)
  en_vocab_path = os.path.join(data_dir, "vocab%d.en" % en_vocabulary_size)
  create_vocabulary(fr_vocab_path, train_path + ".fr", fr_vocabulary_size, tokenizer)
  create_vocabulary(en_vocab_path, train_path + ".en", en_vocabulary_size, tokenizer)

  # 分别为英文与法文的的训练集转换成token_id的形式
  fr_train_ids_path = train_path + (".ids%d.fr" % fr_vocabulary_size)
  en_train_ids_path = train_path + (".ids%d.en" % en_vocabulary_size)
  data_to_token_ids(train_path + ".fr", fr_train_ids_path, fr_vocab_path, tokenizer)
  data_to_token_ids(train_path + ".en", en_train_ids_path, en_vocab_path, tokenizer)

  # 分别为英文与法文的的测试集转换成token_id的形式
  fr_dev_ids_path = dev_path + (".ids%d.fr" % fr_vocabulary_size)
  en_dev_ids_path = dev_path + (".ids%d.en" % en_vocabulary_size)
  data_to_token_ids(dev_path + ".fr", fr_dev_ids_path, fr_vocab_path, tokenizer)
  data_to_token_ids(dev_path + ".en", en_dev_ids_path, en_vocab_path, tokenizer)

  return (en_train_ids_path, fr_train_ids_path,
          en_dev_ids_path, fr_dev_ids_path,
          en_vocab_path, fr_vocab_path)
```

##2.seq2seq_model.py
这个包里是一整个class, 创建了序列到序列的带有注意力机制与多层buckets的模型。
这个buckets的含义之后会讲到。

那么这个class中，将一个多层的循环神经网络作为编码器encoder,然后将一个基于注意力机制的RNN作为解码器decoder。这个模型与这篇论文http://arxiv.org/abs/1412.7449中描述的模型一模一样，更多的细节可以参考论文。

这个class允许使用GRU也可以使用LSTM。GRU是2014年提出来的，将忘记门与输入门合成了一个单一的更新们，同样还混合了细胞状态与隐藏状态和其他一些改定，相对来说比LSTM要简单一点。但两者的效果据说其实区别不大。

这里还使用了sampled softmax来hold住巨大的词表。因为softmax的输出会对每一个输出节点都产生词表大小的向量，向量中的每一个元素都表示该位置上的词的概率。可想而知，如果词表非常大，那个softmax就会输出一个巨大长度的稠密向量。所以这里采用了采样的softmax，对词表进行抽样，使得输出的维度减少。关于sampled softmax可以详细读下这篇文论：http://arxiv.org/abs/1412.2007

单层但是有双向编码的模型详情可以看论文http://arxiv.org/abs/1409.0473

整个class分成三个部分：

###2.1 建立模型
传入的参数：
**source_vocab_size**: 法文字典的大小（法文翻译成英文，所以法文是source)
**target_vocab_size**: 英文字典的大小（英文是输出，故为target)
**buckets**: 一对对（I,O）组成的list。I 表示输入法文句子的最大长度，O表示输出翻译后的英文句子的最大长度。一个样本，如果它的输入或输出超过了这个buckets的I和O的大小，那么这个样本就会到下一个buckets中，下一个buckest会有更大的I和O。而如果样本的大小不足buckets中的大小，那么就自动补全这个句子到buckets规定的长度。这是因为每个句子的输入与翻译后的输出都是各异的，为了统一长度，将样本分到不同规格的桶中，以方便训练。这里我们假设桶buckets是这样升序排列的： [(2, 4), (8, 16)].
**size**: 每一层中的神经元个数
**num_layers**: 神经网络的层数
**max_gradient_norm**: 最大的梯度
**batch_size**: the size of the batches used during training;模型的建立与batch_size是独立的，所以这个大小可以在初始化tf之后改变的。
**learning_rate**: 初始的学习率
**learning_rate_decay_factor**: 学习衰减率。（在训练的后期，需要减小学习率）.
**use_lstm**: 是否使用lstm,设置为true的话，就用lstm代替GRU（默认是GRU)
**num_samples**: 在sampled softmax 中sample的大小
**forward_only**: 只做前向计算，如果设置这个参数，那么就不再做后向计算了。
**dtype**: 存储内部变量的数据类型

```
def __init__(self,
               source_vocab_size,
               target_vocab_size,
               buckets,
               size,
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               use_lstm=False,
               num_samples=512,
               forward_only=False,
               dtype=tf.float32):
    # # 1.赋值各个参数与变量
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(
        float(learning_rate), trainable=False, dtype=dtype)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor) //新的学习率是学习率*衰减率
    self.global_step = tf.Variable(0, trainable=False) //全局的迭代次数，设置初始值为0

    # # 2.创建sample_softmax_loss_function
    # 如果要使用sampled softmax, 那就需要一个output projection（即创建一个output_projection变量）
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax只在采样的size小于整个字典的size的时候有意义，否则就不需要采样了。。。
    if num_samples > 0 and num_samples < self.target_vocab_size:
      # 创建参数变量，在最后的输出层，进入softmax之前的线性函数的参数w,b
      w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
      w = tf.transpose(w_t)
      b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
      output_projection = (w, b)

      # 根据sampled softmax 计算sampled loss
      def sampled_loss(inputs, labels):
        //将[1,2,3]形式的label转变成[[1],[2],[3]]
        labels = tf.reshape(labels, [-1, 1]) 
        # 另外，需要将参数都转换成float32，为了防止数值的不稳定性
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(inputs, tf.float32)
        //返回的是采样后的softmax loss
        return tf.cast(
            tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                       num_samples, self.target_vocab_size), dtype)
      softmax_loss_function = sampled_loss

    # # 3.创建多层神经网络RNN
    # 默认使用GRU，否则使用lstm,输入的size是神经元的个数
    single_cell = tf.nn.rnn_cell.GRUCell(size)
    if use_lstm:
      single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
    cell = single_cell
    # 如果设置的神经网层数大于1，那么就创建多层
    if num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

    # # 4.创建 seq2seq function:
    # we use embedding for the input and attention.
    # 直接调用 tf.nn.seq2seq.embedding_attention_seq2seq这个函数，将需要的参数传入即可
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      return tf.nn.seq2seq.embedding_attention_seq2seq(
          encoder_inputs,
          decoder_inputs,
          cell,
          num_encoder_symbols=source_vocab_size,
          num_decoder_symbols=target_vocab_size,
          embedding_size=size,
          output_projection=output_projection,
          feed_previous=do_decode,
          dtype=dtype)

    # # 5.准备要喂给rnn的输入数据
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    # [-1][0]表最后一行第1列,是最大的encoder输入
    for i in xrange(buckets[-1][0]):  //最后一个桶是最大的
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    # [-1][1]表示的是最大的桶中的输出O的大小
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                name="weight{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    # 训练的输出与损失Training outputs and losses.
    # 如果设置了只需要前向运算：
    if forward_only:
      self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
          softmax_loss_function=softmax_loss_function)
      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        for b in xrange(len(buckets)):
          self.outputs[b] = [
              tf.matmul(output, output_projection[0]) + output_projection[1]
              for output in self.outputs[b]
          ]
    # 否则会进行后向计算，tf.nn.seq2seq.model_with_buckets函数输出的是训练结果与损失两个数据
    else:
      self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets,
          lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function)

    # 随机梯度下降Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.all_variables())
```
###2.2 训练迭代模型
以上已经建立了模型，现在就要将数据喂给模型，并且实行迭代计算。

输入参数：
**session**: tensorflow session to use.tensorflow的会话
**encoder_inputs**: 由一组向量组成list，作为encoder的输入，向量里的元素是numpy int
**decoder_inputs**: 由一组向量组成list，作为decoder的输入，向量里的元素是numpy int
**target_weights**: 一组float类型的向量组成的list,作为target_weight
**bucket_id**: which bucket of the model to use.bucket的唯一编号
**forward_only**: whether to do the backward step or only forward.是否只做前向计算

返回的是：
一个三元组，包括：梯度准则，average perplexity, and the outputs.
 如果使用了后向计算才会有梯度输出；
 perplexity是一个评估模型的指标；
 output就是训练模型的输出。
      
Raises:
如果encoder_inputs, decoder_inputs, or target_weights的长度和与之对应的bucket_id的那个桶大小不同的话会报错

```
def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
   
    # 首先要检查一下大小是否符合，如果不一致就提示错误
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

```
###2.3 获取样本批数据
从特定桶中随机获取一批数据，这批数据是为上面step方法的输入做准备的。
这里，会重新为数据建立索引，为了符合step中的格式。

输入的参数：
传入这样一个数据集，这个数据集的长度是_buckets的长度。我们从原始的数据中抽取样本[Source, Target],放到与之长度最接近的bucket中，然后补全长度。这样最后的数据集的形式是：
[1, list[[S,T],[S,T],[S,T]...[S,T]]
 2,list[[S,T],[S,T],[S,T]...[S,T]]
 3,...
 4,...]
 也就是说，每个桶中会有多对样本，每一对样本都由[Source, Target]组成，而Source和target分别指一个对应的句子分词之后转换成id的list.也就是每一个样本是一句话，Sourse是这句话的法文，Target是这句话的英文。都是以词的id形式存储的。
 
 此处get_batch这个方法其实是从这些桶中随机抽取样本组成一个batch的数据集，来作为模型的输入

返回的是：
一个三元组：(encoder_inputs, decoder_inputs, target_weights) 

```
def get_batch(self, data, bucket_id):
   
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # 随机从桶中获取数据，组成一个batch,如果需要的话补全数据，并且将encoder inpunts逆序排列（据实验验证逆序比顺序效果要好），然后在decoder inputs每句话前加上GO对应的id,表示一句话的开始。
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # 将小于桶大小的样本补全，然后整句话逆序排列
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # 将decode input前面加上GO，然后再补全长度
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights
```

##3. translate.py
这里调用了以上两个包，实现获取数据，预处理数据，建立模型，训练，输出的整个过程。

关于rnn 的机器翻译模型，详细的介绍也可以才能看以下论文：
 http://arxiv.org/abs/1409.3215
 http://arxiv.org/abs/1409.0473
 http://arxiv.org/abs/1412.2007
 
 首先导入需要的包，其中包括了 data_utils，seq2seq_model这两个刚刚自己写的包
```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn.translate import seq2seq_model

```
定义参数,这些参数是等下数据预处理，训练模型要用到
```
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")  //学习率为0.5
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, 
                          "Learning rate decays by this much.")  //学习衰减率为0.99
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,  
                          "Clip gradients to this norm.")  //最大的梯度准则
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")  //一批样本中的样本数
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")  //每层神经网络的神经元个数
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.") //神经网络的层数
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.") //英文字典的大小
tf.app.flags.DEFINE_integer("fr_vocab_size", 40000, "French vocabulary size.") //法文字典的大小
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).") //训练数据的最大数量（0为无限制）
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS
```
定义buckets,以下设置了4个桶，每次有样本进来就到离它大小最相近的那个桶，然后不足的数量就补全。
在seq2seq_model.Seq2SeqModel中可以看到详细的原理
```
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
```
### 获取数据并分桶
获取法文（source)与英文(target)数据,然后把他们放到对应大小的桶中。
这边获取数据就是data_util中已经转换成Token_id的数据，每一行是一句话，并且两种语言的文件中，相同行的句子是一个意思。

输入的参数：
**source_path**: 以分词后的id形式存储的法文数据的路径
**target_path**: 以分词后的id形式存储的英文数据的路径，文件中的数据每一行都是与法文中的每一行一一对应的。
**max_size**:最大读取的行数，如果设置为0或者none的话，就表示全部读取，没有限制。

返回：
**data_set**
这个数据集的长度是_buckets的长度。我们从原始的数据中随机抽取样本[Source, Target],放到与之长度最接近的bucket中，然后补全长度。这样最后的数据集的形式是：
[1, list[[S,T],[S,T],[S,T]...[S,T]]
 2,list[[S,T],[S,T],[S,T]...[S,T]]
 3,...
 4,...]
 也就是说，每个桶中会有多对样本，每一对样本都由[Source, Target]组成，而Source和target分别指一个对应的句子分词之后转换成id的list.也就是每一个样本是一句话，Sourse是这句话的法文，Target是这句话的英文。都是以词的id形式存储的。
```
def read_data(source_path, target_path, max_size=None):
 
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      # 将source 和target一行一行读进来，两者的行与行之间是意思一一对应句子
      source, target = source_file.readline(), target_file.readline()
      # 计数（总共读了多少行）
      counter = 0
      # 对每一行做遍历
      while source and target and (not max_size or counter < max_size):
        # 每开始处理一行就计数为1
        counter += 1
        # 每100000行的时候就打印一下行数
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        # 分别对source和target对应的那行按照空格分隔，然后将每个元素转换成int类型
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        # 然后在target_id后面追加EOS_ID（2），表示一句话的结束
        target_ids.append(data_utils.EOS_ID)
        # 然后对每一个桶做遍历，看看这句话放进哪个桶里比较好
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          # 如果source和target的长度都小于桶的大小，那么就往这个桶中追加进这组样本，然后停止遍历
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        # 再读一行新的数据进来
        source, target = source_file.readline(), target_file.readline()
  # 返回data_set
  return data_set
```
### 创建模型
现在创建rnn翻译模型，并且在tf.session中初始化和加载参数

输入的参数就两个：
**session**是tensorflow的会话
**forward_only**表示只进行前向计算，如果设置了这个参数，就表示不进行后向计算。

```
def create_model(session, forward_only):
 
  # 数据类型
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  # 创建模型
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.en_vocab_size,
      FLAGS.fr_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
  # 检查点
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  //如果路径已经存在，则打印提示，然后将这些参数读入作为模型的初始参数
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  //如果不存在，则初始化初始参数
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  # 返回这个rnn模型
  return model
```

### 训练模型
使用WMT数据训练英文-->法文的翻译模型。
```
def train():
  
  # 准备 WMT 数据，将分词并且转换成token-id的训练集与测试集保存在 FLAGS.data_dir路径中
  print("Preparing WMT data in %s" % FLAGS.data_dir)
  en_train, fr_train, en_dev, fr_dev, _, _ = data_utils.prepare_wmt_data(
      FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.fr_vocab_size)

  # 创建session,执行一系列操作
  with tf.Session() as sess:
    # 1.创建模型
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    # 2.获取数据并放到buckets中，然后计算每个buckets大小
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(en_dev, fr_dev) //测试集分桶
    train_set = read_data(en_train, fr_train, FLAGS.max_train_data_size) //训练集分桶
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))] //测试集的每个桶计算样本数量
    train_total_size = float(sum(train_bucket_sizes)) //测试集总共的样本个数

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    # 每个桶中的样本数量占总数量的比例
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # 3.开始循环训练.
    step_time, loss = 0.0, 0.0  //循环的时间与该次循环的损失
    current_step = 0  //目前的循环次数
    previous_losses = []  //记录历史损失
    while True:
      # 3.1 选择一个桶，这里我们是随机选择[0,1]之中的随机数，然后在train_buckets_scale中找出比这个随机数大的最小比率的那个桶（好绕口。。）
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # 3.2 获取一批数据并且执行一次循环
      start_time = time.time()  //当前时间为开始时间
      # get_batch是从指定的bucket_id中随机获取一批样本
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      # 输出执行这一步的损失
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      # 输出这一步的时间
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      # 将这一步损失追加到loss中
      loss += step_loss / FLAGS.steps_per_checkpoint
      # 循环次数加1
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      # 在一定步数的时候：
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.打印损失值
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # 如果前三次迭代中模型没有提升（损失没有减少）就减少学习率
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # 保存检查点的参数，然后将时间与损失归0
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # 使用测试集并输出测试集的损失
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              "inf")
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()
```

###解码

```
def decode():
  with tf.Session() as sess:
    # 创建模型并加载参数
    model = create_model(sess, True)
    model.batch_size = 1  # 每次都解码一句话

    # 加载两个语言的字典
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.en" % FLAGS.en_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.fr" % FLAGS.fr_vocab_size)
    en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    # 从控制台输入一句话
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # 将这句话转换成token-id
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
      # 这句话属于哪个桶呢？选出适合的bucket_id
      bucket_id = len(_buckets) - 1
      for i, bucket in enumerate(_buckets):
        if bucket[0] >= len(token_ids):
          bucket_id = i
          break
      else:
        logging.warning("Sentence truncated: %s", sentence) 

      # 将输入的句子作为只有一个元素的batch，用于模型的输入
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # 获取这句话经过模型之后的logit输出
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # 因为输出的是每个词的概率，所以取出logit的最大值所在的id（根据这个id去找字典中的word)
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # 如果有 EOS 在输出中，就在那里阶段这句话，因为EOS表示一句话的结束.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # 根据id在词典中找到对应的法文，然后打印出这句翻译好的句子
      print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()
```
###测试翻译模型
```
def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)
```

本文只是根据自己的理解对代码进行简单的标注与解释，存在的错误与纰漏，日后会及时更新。