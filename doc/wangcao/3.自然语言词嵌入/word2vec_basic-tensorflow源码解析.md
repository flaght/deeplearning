
## word2vec Tensorflow 基础案例

这是来自tensorflow官网的一个小案例，隶属于2.7节 字词的向量表示，官方文档中有对理论的讲解，可参见以下链接：
http://www.tensorfly.cn/tfdoc/tutorials/word2vec.html

代码的出处可参见github：https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

word2vec是利用深度学习来建立的词向量的表示，主要有2种方式：CBOM,Skip-gram。上面这个链接中着重讲了skip-gram，若没有基础，想详细了解词向量的知识，欢迎参见笔者之前整理的通俗文档，https://www.zybuluo.com/wangcao/note/473832
笔者才疏学浅，承蒙受阅，自当恩宠，若不当之初，跪求指出。

下面的案例，是针对skip-gram给出的.

### 导入需要的库


```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
```

### step 1 获取数据

首先是需要从网上将数据的压缩包读取下来。方法maybe_download中需要传入两个参数，分别是filename数据的压缩文件名，和expected_byte期望获取的字节长度。

当然，会去检验之前是否已经运行过程序下载了这个数据，若本地没有这个文件。

os.stat() 方法用于在给定的路径上执行一个系统 stat 的调用。具体细节可见http://www.runoob.com/python/os-stat.html
st_size: 表示文件以字节为单位的大小；包含等待某些特殊文件的数据。
所以如果statinfo.st_size == expected_bytes，那么就表示实际上读取的数据大小和预期的是一样，所以读取成功。否则的话，就打印实际大小，然后爆出异常表示文件无效了。

最后返回的是读取到的数据。


```python
# 数据获取的网址
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
# 如果文件不存在，则去获取
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

# 调用以上方法获取文件名为text8.zip的数据，字节长度为31344016
filename = maybe_download('text8.zip', 31344016)
```

上面读到的数据是一个压缩文件，我们需要去解压文件，然后将数据变成string类型的list格式。
要获取的是zip文件中的第一个文件，.as_str表示将其转为string,后面调用split
最后输出的是一个list of words


```python
def read_data(filename):
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

# 调用以上方法获取词列表
words = read_data(filename)
# 打印一下词的个数看看
print('Data size', len(words))

```

### step 2 建立词典


首先定义词典的大小是50000.

build_dataset这个方法是将词转换成具有对应编码的字典，对每一步以下都做了尽量详细的解释：

输入的words，就是上一步中获取的词列表。

1.创建一个叫count的变量，是一个list里面嵌套一个list，放的是'UNK', -1，前者表示词，后者表示它出现的频数，等等要把所有词与频数都一一放进这个list中。

2.count.extend就是追加词与频数了，collections.Counter(words)表示的是将word中的所有词进行计数，变成[('yes', 27), ('no', 23), ('to', 12)]这样的形式，并且默认是降序排列的。.most_common(n)表示的是取出最高频的前n个数。我们先前定义的词典大小是50000，所以排名之后的词因为出现太少就无需放入了。

3.创建一个空的字典dictionary，等等用来装{词，词编码}的。

4.然后对count中的数据（词，频数）做遍历，将{词， 目前字典的长度}添加进字典。目前字典的长度可以看做是该词的编码。比如第一个词W1，这是字典是只有，所以长度为0，那么w1的编码就为0，输入{w1,0},同理，之后会依次进入{w2,1},{w3,2}...如此每个词都有一个自己的编码数字了。

5.现在再来创建一个空的list.
并且创建一个叫data的变量，赋值为0先。

6.接着对words中的词做遍历，如果这个词在我们刚刚建的字典里，那么它的索引就是这个词在字典中的value,好拗口，索引就是词的编码。
如果这个词没有被选中在字典中，那么索引就是0了，也就是dictionary['UNK']。然后就往unk_count这个变量上计数
之后，将所有的索引都追加到刚刚建的data列表中。

7.对count的第1行第2列赋值为刚刚计算的出现索引为0的总数，代替了原来的-1.也就是这一行表示稀少的词有多少个（50000名以外的为稀少词）

8.之后，将原来的字典做了一个转换，把原来的key变成value,把原来的value变成key,最后变成了[编码，词]的形式。

9.build_dataset返回的是
```
data: list[index],list中放的是所有词对应的编码
count: list[[word, count]],放的是每个词及其出现的频数
dictionary: dic{word, index},词与对应的词编码
reverse_dictionary:dic{index, word},是词编码对应的词
```



```python
# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
    
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
    
  count[0][1] = unk_count

  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
  return data, count, dictionary, reverse_dictionary

# 调用以上方法
data, count, dictionary, reverse_dictionary = build_dataset(words)
# 将words这个词列表删除，因为我们已经将他转换成我们需要的数据，而它也不再需要被使用，为了减少内存就删除了，真可怜，感觉是被利用完了抛弃的。
del words  
# 既然大费周章，那有必要看看处理完的数据是否如预期，取出count中前5的数据
print('Most common words (+UNK)', count[:5])
# 再取出前10的编码，并且根据这个编码在字典里取出相应的词
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0
```

### step 3 生成training batch

这一步，我们要为skip-gram model的输入做准备了。目的是生成一批样本数据，与一般的神经网络案例一样，样本由特征与标签组成。只是这里的特征是指一个窗口词中的中间词，标签是对应这个中间词前后的词。根据skip-gram的原理我们很容易明白，现在是要用中间词去推测上下文的词，那么当然中间词相当于特征需要被输入到神经网络中。

为了便于理解下面这个方法到底在干啥，这里先事先说明方法最终输出的结果是怎么样的：
输出的是两个变量：batch, label。batch大小是事先设定的batch_size的大小，这里设定的是8，label的大小是[batch_size, 1]
将batch打印出来是：
```
[3084 3084 12 12 6 6 195 195]
```
将label打印出来是：
```
[[5239]
 [12]
 [6]
 [3084]
 [195]
 [12]
 [6]
 [2]
```
这两个数据是什么意思呢？意思是，例如batch中词编码是3084，它的上下文的词的编码是5239,12.或者说3084->5239,3084->12.

输入的参数是
```
batch_size：每批数据的样本数量
num_skips:实际用于预测（也就是用于负采样）的词数。全部利用的情况下等于2 * skip_window, 并且受限于 num_skips * num_history = batch_size, 要满足batch_size % num_skips == 0
skip_window:取左右单词的个数。窗口的大小为2*skip_window+1
```

**代码块解释：**

1. assert表示的是终止，如果不满足后面的条件则终止运行。
首先batch_size必须是num_skip的倍数，因为num_skip是对每个中心词的采样，而所有中心词的采样数应该等于batch_size,假设中心词我们叫他histoty_word,那么肯定满足公式history_words * num_skip = batch_size.
第二个条件，之前说了num_skip是负采样数，那么肯定小于总的样本数，而对于一个样本，总的样本数就是skip_window * 2(也就是中心词的上下文词）


2. 变量buffer
buffer是用来缓存从一个窗口中挖出来的词的，比如窗口长度为3，采样为2，文本是'我/是/中国/人/我/真/骄傲/啊'。首先buffer会装进[‘我’，‘是’，‘中国’]，然后通过接下去的一系列转换，把[是, 是]放进bacth，把上下文[我],[中国]放进lebel.这表示已经产生了一组样本（其中包括2个样本）。同理，然后buufer会变成[‘是’，‘中国’，‘人’]，接下来就是对‘中国’为中心词提取两边的上下文词了。
可见，buffer的作用就像一个小窗口，不断地在文章中从头往尾地一个词一个词地移动，每个词都会被当成中心词，并提取对应的上下文词。
不过注意的是，方法中所有的操作都是针对词编码的，记得step2中，我们输出了一个data变量，里面放的就是与原文词顺序相对应的词编码。
全局变量data_index初始值为0，在n次循环之后（for _ in range(span)），buffer中就追加进了最前面3个词的编码，buffer的长度永远都是窗口大小，如果后面再追加进东东，那么前面的东东就会被移除，这是deque的其中一个特性，真的好方便啊。


3. 双斜杠表示整数除法
新的Python3中除法“/”变成了浮点除法，Python开发者为了保留老版的整数除法，就用双斜杠代替了



```python
def generate_batch(batch_size, num_skips, skip_window):
  global data_index

# 若不满足后面的条件则终止程序
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window

# 创建放batch的数组，创建放label的数组，大小是(batch_size * 1)
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

# 窗口大小（2 * skip_window是上下文词的长度，+1表示算上了中心词自己）
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]

# 用来缓存一个窗口的词编码(长度小于等于窗口大小)
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data) #到数据尾部后跳回数据头部重新开始，一种环形的结构

# 针对buffer中的词，提取中心词与上下文，分别追加到batch与label中
  for i in range(batch_size // num_skips):
    target = skip_window  # target为中心词所在的索引
    targets_to_avoid = [skip_window]
    
    # 做循环（循环的词数是num_skip的值，也就是说要多少个采样，就循环多少次去采）
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1) # 这里随机取未预测过并且非history的词
      targets_to_avoid.append(target)  # history和已取作skip-gram的词
      batch[i * num_skips + j] = buffer[skip_window] # 将中心词放到batch中
      labels[i * num_skips + j, 0] = buffer[target] # 将随机取出的某个上下文词放到label中
    # 在buffer后追加下一个词的编码，因为长度是固定的，所以第一个词被挤掉了，也就是窗口向后移了一个位置。
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
    
  return batch, labels

# 调用以上方法获取batch,label
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
# 打印看看
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
```

打印内容为：
```
3084 originated -> 12 as
3084 originated -> 5239 anarchism
12 as -> 3084 originated
12 as -> 6 a
6 a -> 12 as
6 a -> 195 term
195 term -> 6 a
195 term -> 2 of
```

### step 4 创建并训练a skip-gram model.

定义一些之后需要用到的常量。

首先是模型的一些参数，这个无需解释。

另外是验证集的参数，意思是要从字典里取出一些词来当验证集，因为字典里的词是按照频数降序的，所以这边取最前面的一些词。



```python
batch_size = 128      # 批处理的样本数量
embedding_size = 128  # embedding vector的维度
skip_window = 1       # 上文或下文的词数，单指一边，上下文总长度为skip_windows * 2
num_skips = 2         # 负采样的数量，也是实际预测的数量

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # 取字典中的16个词做验证集
valid_window = 100  # 这16个词是从字典的前100个id的词中随机无放回的取出的
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # 16个词的话对应的上下文有16*2*2=64,所以验证集有64个样本
```

创建模型

利用梯度下降法求损失最小的时候的嵌入矩阵（这边的嵌入矩阵是指上文说的投影矩阵）。

为了看看词嵌入矩阵的质量如何呢，还特地选出来16个词组成了验证集，根据我们得到的嵌入矩阵去计算与这16个词与字典中所有词的相似度，方便之后根据相似度取出最相似的词来看看效果。


```python
graph = tf.Graph()

with graph.as_default():

  # Input data.设置占位符
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # 创建变量：投影矩阵
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # 让输入的词向量与投影矩阵相乘变成词嵌入矩阵
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    # 词对应LR分类器的theta和biase参数
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # 调用nce_loss计算nce损失，并计算所有损失的均值，每次计算损失，tf.nce_loss都会自动选择新的负标签作为样本
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                     num_sampled, vocabulary_size))

  # 使用SGD优化器，学习率为0.1
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  # 得到normalized 投影矩阵
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
 
  # 获取验证集的词嵌入矩阵
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)

  # 计算验证集和所有词典中词的cosine相似度，后期可以观察词向量的质量变化
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # 初始化所有变量
  init = tf.initialize_all_variables()
```

### step5 开始训练

训练的目的，也是整个模型的目的呢是为了得到词嵌入矩阵，也就是我们常说的词向量。这个词向量不再是传统的one-hot词向量，而是word embedding词向量。在向量中可以计算词与词的相似性。所以最后一步，提取了与验证集中的词最相似的前8个词，可以用来观察模型的质量好坏。


```python
# 训练的步数为100000
num_steps = 100001

with tf.Session(graph=graph) as session:

# 首先执行初始化所有变量
  init.run()
  print("Initialized")

# 平均损失初始值为0
  average_loss = 0

# 开始100000次循环
  for step in xrange(num_steps):
    # 获取batch 与 label的数据
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    # 准备好要喂的数据，也就是将获取的数据放在盘子里，等等直接把盘子给模型的输入
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    # 每循环一次都得到最优值，损失，然后将损失累加在average_loss上
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    # 每2000次的时候，打印一下这2000次内的平均loss
    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # 注意注意，下面的东东很耗资源滴，如果你把10000改成500的话速度回满20%哦
    if step % 10000 == 0:
      sim = similarity.eval()  # 计算相似度
      # 对验证集做遍历，取出每一个词作为中心词
      for i in xrange(valid_size):
        # 取出每一个词作为中心词，取出与这每个词相似度最高的8个词，打印出来看看效果如何
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
# 得到最终的词潜入矩阵
  final_embeddings = normalized_embeddings.eval()
```

### 可视化embedding


```python
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
```
