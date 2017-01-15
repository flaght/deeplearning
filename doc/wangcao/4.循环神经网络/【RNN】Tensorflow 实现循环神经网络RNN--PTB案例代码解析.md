# 【RNN】Tensorflow 实现循环神经网络RNN--PTB案例代码解析

标签： 王小草深度学习笔记

---

时间：2016年10月26日
笔记整理者：王小草

一大早雨下得认真狂野，坐上了5倍价钱于常日的滴滴在100米中堵了15分钟，只为到地铁起点站去拥有一个小时的限量版座位。前几周我是看推理小说度过地铁时光的，最近发现，看代码其实也蛮好。到公司的时候高跟鞋变成了高跟船，但我心里想着的是今天要把tensorflow的RNN案例测一遍。也许我离程序猿的世界更近了，希望这是我把这世界带进了生活，而不是生活走进了这世界。

将tensorflow的RNN测一遍其实就是在命令行几句代码的事，我想有必要将代码拿出来好好读一遍。于是本文的主要内容就是对tensorflow官网RNN案例的代码解析。

RNN的理论讲解请见《王小草【深度学习】笔记第六弹》，另外《神奇循环神经网络RNN-python代码解析》一文讲解了如何用python去实现RNN的每个步骤。然而实际的生产环境中，数据挖掘师们可能不会自己手动去完整地写一个神经网络，因为目前有许多非常好的深度学习的框架，安装这些框架，只需要几句代码，就能搭建一个复杂的神经网络，并且可以根据自己需求调节神经网络的结构与参数，如此方便时尚，何乐而不为呢？本文中的tensorflow就是这些框架之一。

tensorflow RNN 案例 源码地址：
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/
整个程序由两部分组成：
1.获取数据与预处理 reader.py
2.RNN模型搭建与训练  ptb_word_lm.py 
接下来会分别对两者进行解释。

## 1.reader.py
tensorflow官方文档中循环神经网络的案例使用了Penn Tree Bank (PTB) 数据集。数据集的下载链接是：
http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tg

该数据集已经预先处理过并且包含了全部的 10000 个不同的词语(也就是语料库大小维10000)，其中包括语句结束标记符，以及标记稀有词语的特殊符号 "< unk >" 。我们在 reader.py 中转换所有的词语，让他们各自有唯一的整型标识符，便于神经网络处理。

下载下来的是一个tar压缩包，里面包含三类数据，分别是929000训练样本，73000验证样本，82000测试样本。
总共有10000个词在词典中。

本文要讲解的是在下载好以上这个数据集之后，如何获取，预处理的过程。最后的输出应是可以直接被lstm模型使用的。

###1.1 首先导入需要的包


```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
```

###1.2 数据预处理
根据文件名将数据读进来，然后decode根据utf-8解码，然后将换行符换成< eos >,最后根据空格分隔（相当于对一整片文章进行分词），形成了一个大的list


```python
def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().decode("utf-8").replace("\n", "<eos>").split()
```

###1.3 建立词典
建立词典的过程其实就是：
对每个词计数，生成（词，词频）--->根据词频排序，根据排序取处前n个词--->去掉词频保留list[词]--->对词建立索引，生成dic{词，索引}


```python
def _build_vocab(filename):
 
  # 根据文件名读进数据并分词
  data = _read_words(filename)

  # 对出现的words进行统计，counter函数返回的是（词，词频），并且默认是降序的
  counter = collections.Counter(data)
  # conuter.items()会返回一个（key, value)的tuple.-x[1]表示按照第而个元素降序，x[0]表示按照第以个元素升序。 也就是说按照词频数降序，词频相同的话就按照词来升序排列
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  # 感觉这个像unzip 就是把key放在一个tuple里，value放在一个tuple里，排完序之后只需要保留词就行
  words, _ = list(zip(*count_pairs))
  # 将每个词编码，建立唯一的索引
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id
```

###1.4 根据词典，将数据中所有词替换成对应的索引
如果样本中的词在词典中存在，那么就将它替换成对应的索引。
```
def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]

```

###1.5 调用以上方法预处理原始数据，并输出训练集，测试集，验证集
 传入的参数是data_path,是原始数据解压的那个文件的路径。
 输出的是四个数据：训练集，测试集，验证集，字典的长度

```
def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """
  # 三个数据的路径
  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  # 调用 _build_vocab方法，根据训练集中的数据建立词典
  word_to_id = _build_vocab(train_path)
  # 分别读取三类数据，并将他们预处理后转换成id形式
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  # 字典的长度
  vocabulary = len(word_to_id)
# 返回三个数据集与字典的长度
return train_data, valid_data, test_data, vocabulary
```

###1.6 生成批样本
对以上处理好的数据集进行迭代，将数据分割成多批样本，然后将这一批批样本以tensor的形式返回

输入的参数是：
raw_data:是从ptb_raw_data中输出的某一个数据集
batch_size:每批样本的大小
num_steps:每个样本中的词数
name:这次操作的名称（可有可无）

返回的是：
一对tensor数据，一份是作为自变量x，一份是与之对应的应变量y。
他们的数据维度都是[batch_size, num_steps],每一行都是一个样本，总共有batch_size个样本，每个样本中有num_step个词。

这样的话，RNN循环的次数其实就是num_step词的个数。

```
def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.
  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).
  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    # 二话不说，先输入的数据变成tensor
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    # 数据集中的词数（比如这个数据集是指一整个训练集，是一个list[word_id]）
    data_len = tf.size(raw_data)
    
    # 接下来是把一整段的文本转换成一批样本数据x和y，然后返回
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
    y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
    return x, y
```

## 2.ptb_word_lm.py 
###2.1 首先导入需要的包
```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn.ptb import reader
```

###2.2 设置参数

括号里面有三个元素：第一个是参数的名称，第二是参数，第三个是参数的解释。当我们在命令行运行模型的时候，可以从外部传入着三个参数。

data_path 与 save_path这两个参数无须解释。
model 这个参数指的是需要运行的模型的大小，有三个类型可以选择：small,medium, large，选择不同的类型那么模型的配置与规模也是不同的。关于这些模型的配置，在接下去的代码中会讲到。在本次案例中，我们选择的是small.
```python
flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS
```

###2.3 获取数据类型的方法
根据传入的参数返回float16或float32

```python
def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32
```

###2.4 获取数据的类
创建一个 PTBInput类，用来获取数据，并且定义与数据相关的一些参数与属性

data指的是词id的列表，就是reader.py中ptb_raw_data方法中返回的某个数据集

在def中定义了这些属性：
batch_size 一批样本中的样本数，在语言模型中表现为一组样本中有几个句子
num_steps:单个样本总的词数
epoch_size:迭代的次数？
input_data和targets分别是一批样本数据中的x和y


```python
class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)
```

###2.5 创建lstm模型的类

input_是输入数据的一个类，应该是2.4中的PTBInput
config是配置函数，应该是根据传入的参数small来调用对应的配置

首先定义了一个lstm的cell，这个cell有五个parameter，依次是
number of units in the lstm cell 隐藏层的个数
forget gate bias  忘记门的偏执项
input_size 输入数据的维度
state_is_tuple=False
activation=tanh 激活函数

这里我们仅仅用了3个parameter,即size，也就是隐匿层的单元数量以及设forget gate
的bias为0，还有state_is_tuple=True. 如果把这个bias设为1效果更好，虽然会制造出不同于原论文的结果。

```python
class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._input = input_
    
    # 参数：训练样本的参数与模型的参数
    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    # 定义LSTM_cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    
    # 如果属于训练并且输出的保留概率小于1时，每一个lstm cell的输入以及输出加入了dropout机制
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    
    # 这里的cell其实就是一个多层的结构了。它把每一曾的lstm cell连在了一起得到多层的RNN
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
     
    # 隐匿层的初始值是设为0，大小为batch_size
    self._initial_state = cell.zero_state(batch_size, data_type())

    # 设定embedding的变量，并且转化输入单词为embedding里的词向量（embedding_lookup函数）
    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    # 如果在训练并且输出的保留概率小于1时，那么对输入也进行dropout
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = [tf.squeeze(input_step, [1])
    #           for input_step in tf.split(1, num_steps, inputs)]
    # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
    
    # 从state开始运行RNN架构，输出为cell的输出以及新的state.输出会被追加到output中，state是不断被更新
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    # 输出定义为cell的输出乘以softmax weight w后加上softmax bias b. 这被叫做logit
    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    
    # loss函数是average negative log probability, 这里我们有现成的函数sequence_loss_by_example来达到这个效果。
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size  # 平均每个样本的损失
    self._final_state = state

    if not is_training:
      return

    # 设置学习率变量
    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    # 根据张量间的和的norm来clip多个张量
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    # 用之前的变量learning rate来起始梯度下降优化器。
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    
    # 训练模型
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())
    
    # 新的学习率
    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
```
###2.6 模型的配置
3 个支持的模型配置参数："small"， "medium" 和 "large"。它们指的是 LSTM 的大小，以及用于训练的超参数集。


```python
class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
```

测试模型的参数


```python
class TestConfig(object):
  """Tiny config, for testing."""
 
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
```

###2.7 运行模型
根据给出的数据运行模型

```python
def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
 
  start_time = time.time()  # 当前时间
  costs = 0.0  #初始化损失为0
  iters = 0  # 初始化迭代次数为0
  state = session.run(model.initial_state)  #运行初始化state

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)
```

###2.7 获取配置
上面定义了4个class：："small"， "medium" ，"large"，"test",根据需要获取配置


```python
def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
```

###2.8 main函数


```python
def main(_):
 
  # 如果没有数据路径（训练数据或测试数据），就报异常
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
 
  # 获取数据（根据输入的datapath获取该路径下的三个文件数据；训练，测试，验证集）
  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data
 
  # 获取相对应的配置，根据Flag中定义的small model，获取smallconfig,然后将batchszie和numstep改成1
  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    # 初始化
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      # 获取数据
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      # 创建模型
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      # 可视化
      tf.scalar_summary("Training Loss", m.cost)
      tf.scalar_summary("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.scalar_summary("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)
```
##3.运行模型

运行方式为在ptb_word_lm.py的文件夹下输入python ptb_word_lm.py --data_path=/tmp/simple-examples/data/ --model small

