# 【RNN】神奇的循环神经网络RNN--Python代码解析

标签（空格分隔）： 王小草深度学习笔记

---

笔记时间：2016年10月25日
笔记整理者：王小草

早上表示上海大晴的天气预报害我现在哆嗦着裙子以下的冰腿码着代码。恩，这么说，2016年的冬天也许是真的来了。怕是又赶不及在初寒之前去一次不拥挤的深秋的旅行。

所以，今天的笔记要写的是循环神经网络，
时间逝往，循环周而复始，新的输入与旧的记忆一起融汇，奔向下一个冬季。
像极了我周而复始的期待，与复始周而的前行。

循环神经网络，英文全称Recurrent Neural Network，简称RNN。
对于RNN的理论与原理，在《王小草【深度学习】笔记第六弹》中有详细介绍，在此不再赘述。本文的的主要内容是用Python来完整得实现一个RNN语言生成模型的案例。包括数据的读入，预处理，前向计算，损失函数计算，反向传播，预测等步骤的详解。

当然，在实际的实现中，有许多现有的优秀的框架，调用这些框架，我们只需要写几句代码就能实现一个神经网络。但是在使用框架之前，完整得理解一份RNN从头到尾的实现代码，也许有助于我们对知识的深入理解，然后，再去使用其他框架，应该是锦上添花的。

## 案例背景
在《王小草【深度学习】笔记第六弹》一文中已经见识过RNN生成模型的厉害了，它可以模仿写小说，代码，诗歌，论文，甚至还可以写出奥巴马的演讲稿。其实最让我对RNN刮目相看的还是它对小四文字的模仿（这个案例来自于寒小阳的博客），那文笔，那文风，作为文艺女青年的我也是甘拜下风啊！

本文案例要让循环神经网络与学习的是从reddit(类似百度贴吧）中爬取的15000条长评论。目的是要让RNN在学习之后也能模仿人类评论者写出一段话。
（本文的案例原文来自于Recurrent Neural Network Tutorial,Part2 - Implementing a language Model RNN with Python, Numpy and Theano.并且参考了韩小阳大神的部分翻译与解释。）

我们先导入需要的包
```
import nltk
import itertools
import csv
import numpy as np
import math
from sklearn.linear_model.logistic import *
import sys
from datetime import datetime
```
下载nltk模型数据，nltk是一个自然语言处理的库，在预处理中会被用来做分句和分词。这个download第一次运行的时候会去下载东东，之后运行都无需下载了。下载有点慢，我翻墙后也下了几乎1个小时。如果没有加这句话的话，会报错。
```
nltk.download("book")
```
创建并设置全局变量，这些变量先不解释，看下去自然会明白的。
```
vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
```

## 数据预处理
爬下来的数据是以csv格式存储的，每一行是一段评论。在被模型所食用之前，需要对数据做预处理。分成以下几步：

###1.分句，并设置句子的开头与结尾
首先用csv.reader将既定目录下的数据文件读进来。

然后对读进来的每条数据进行分句。
因为读取的文件是每行一段评论，一行一行读进来，就是n*1维的，所以每一行的文本就是列索引维0处，表示为x[0].
将x[0]根据utf8解码，然后再调用.lower()函数转换成小写。
nltk.sent_tokenize(a)表示对字符串a进行分句。
intertools.chain((a,b),(c,d))返回的结果是(a,b,c,d).感觉有点想spark中的flatMap

分句之后就对每个句子的前后都加上"SENTENCE_START"和"SENTENCE_END"作为特殊的开始与结束的标志。

```
print "Reading CSV file"
with open('/home/cc/data/RNNlm/reddit-comments-2015-08.csv', "rb") as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # # 对文本按照句子进行分割
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # # 对句子前后加上"SENTENCE_START"和"SENTENCE_END"
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Parsed %d sentences" % (len(sentences))

```
###2.分词
对每个句子做分词。nltl.word_tokenize()可以对英文句子做分词，但中文不行。网上查了下，好像很多都是用jieba做中文分词的（调用jieba.cut()）。
```
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
# tokenized_sentences = [jieba.cut(sent, cut_all=False) for sent in sentences]
```

分完词之后数据的格式应该是[[ ],[ ],[ ]...,[ ]]list中嵌套list,每个小list都是一句话中的词语。打印出来是这样的：
```
[[u'SENTENCE_START', u'i', u'am', u'a', u'girl', u'.', u'SENTENCE_END'], 
[u'SENTENCE_START', u'i', u'want', u'eat', u'rice', u'.', u'SENTENCE_END'],
[u'SENTENCE_START', u'i', u'am', u'a', u'boy', u'.', u'SENTENCE_END'], 
[u'SENTENCE_START', u'i', u'am', u'a', u'boy', u'.', u'SENTENCE_END']]

```
###3.取出高频词，做词向量的映射
我们设定语料库的大小是8000，将词频前8000的数拿出来，然后建立每个词的编码，形成（词，编码）的映射字典

(1) 同样使用itertools.chain这个方法来讲分词后的句子变成一整个大的词List。也就是说将原来的[[a,b],[a,c],[b,c]...,[e,d]]变成[a,b,a,c,b,c...,e.d]
然后对词列表做调用nltk.FreDist方法做词频统计。

(2) 调用most_common(n)方法取出词频最高的前n个词。vocabulary_size表示的是预料库的大小，最前面定义了大小是8000.我们将8000名以外的词都作为低频词，用UNKNOWN_TOKEN来代替，于是UNKNOWN_TOKEN作为一个新词放在语料库中，对它预测和其他词是一样的。所以我们要选出的是前8000-1个词，因为要留一个位置给UNKNOWN_TOKEN。

(3)因为vocab是（词，频数）的结构，我们已经根据频数选出了语料库中的词，这个频数已经没有用了，所以index_to_word去掉了频数，变成了7999大小的词列表;接着将unknown_token追加到预料库中，大小正好是8000.

(4)word_to_index是对每个词都建立一个索引。enumerate(words)会生成(index, wordi)的对应索引，dict里面只是将（i,w)掉了一下顺序变成(w,i)。dict是字典的格式。

```
# # 统计每个词的词频
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

# # 取出高频词,建立索引
vocab = word_freq.most_common(vocabulary_size-1)  # 挑出7999个最高频打词
index_to_word = [x[0] for x in vocab]  # 去掉后面的计数，保留词
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
```

###4. 标注低频词
注意第3步中只是为了去建立一个带索引的词典，训练数据集还是来自与第2步中分完词的tokenized_sentences。
所以对tokenized_sentences中的每句话做遍历，如果这个词在语料库中存在，则返回原词，如果不存在，则返回unknown_token代替原来那个词

```
# # 把词表外的词标注维为unknown_token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in index_to_word else unknown_token for w in sent]
```
###5.生成训练集
现在是所有预处理好的句子了，要把每个句子都变成一组样本，如：
x:[START_TOKEN,W1,W2,W3,W4]
y:[w1,w2,w3,w4,END_TOKEN]
如此前一个词与它后面的词就形成了一一对应的关系了。
注意，所有词都换成索引做之后的运算。
```
# # 取出一句话中的第1个词到最后第二个词为训练样本，取出第2个词到最后一个词为对应的标签，词都是索引表示的。
x_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
print x_train
print y_train
```
打印出来如下：这边取了总共4句话，每句话对应一个x和一个y。
```
[[ 3  0  5  4  8  2]
 [ 3  0  7 10  9  2]
 [ 3  0  5  4  6  2]
 [ 3  0  5  4  6  2]]
 
[[ 0  5  4  8  2  1]
 [ 0  7 10  9  2  1]
 [ 0  5  4  6  2  1]
 [ 0  5  4  6  2  1]]
```

##明确RNN网络结构
回顾一下RNN的结构，如下图：

![2016-10-25 17-59-36屏幕截图.png-43.6kB][1]

U是每个时间t输入xt的权重，V是隐层到输出层ot的权重，W是前一时刻的隐层st-1到st的权重。这些权重是我们最终需要估计出来的。

每次进入RNN的是x_train中的一个句子，比如[START_TOKEN,W1,W2,W3,W4],然后对句子中的每个词作为对应时刻的输入，在t1时刻输入的是START_TOKEN,t2时刻输入的是w1.每个词输入的并不是之前建立的编码，而是该词的one-hot编码，在本案例中，词表大小是8000，所以输入的每个词都是是1*8000维的词向量，词向量中只有该词索引的地方是值为1，其他位置值为0.

本案例中设置隐藏层个H=100，这里需要提一下的是，我们之前也说到了，这个隐状态有点类似人类的“记忆体”，之前的数据信息都存储在这里，其实把它的维度设高一些可以存储更多的内容，但随之而来的是更大的计算量，所以这是一个需要权衡的问题。

输出层也是一个1*8000维的向量，每个维度上都是一个概率，表示，如当输入的词为w1时，下一个词分别可能是这8000个词的概率。

如前所提，我们假设词表大小C = 8000，而隐状态向量维度H = 100。
其中U,V和W是RNN神经网络的参数，也就是我们要通过对语料学习而得到的参数。所以，总的说下来，我们需要2HC+H2个参数，在我们现在的场景下，C=8000，H=100，因此我们大概有1610000个参数需要预估。同时，参数的大小，其实也给我一些提示，比如可能需要多少计算资源，大家可以做到提前心里有个数。我们这里大部分计算其实还好，因为one-hot编码后的矩阵，和其他矩阵相乘，其实可以看做挑选出其中的一列(行)，最大的计算量在Vst处。这也就是说，如果计算资源有限，咱们最好把词表的规模限制一下，不要太大。



##初始化参数
先定义一个类 RNNNumpy，并且初始化参数:U,V,W。我们不能直接把他们都初始化为0，这样在计算过程中会出现"对称化“的问题。初始化的不同对于最后的训练结果是有不同的影响的，在这个问题上有很多的paper做了相关的研究，其实参数初始值的选定和咱们选定的激励函数是有关系的，比如我们这里选tanh，而相关的论文推荐我们使用[−1n√,1n√]之间的随机数作为初始值，其中n是和前一层的连接数。额，看起来有点复杂哈，不过别担心，通常说来，你只要初始化参数为很小的随机数，神经网络就能正常训练。（关于初始化参数的设置请看《王小草【深度学习】第三弹》）

以下是对应的python代码，其中word_dim是词表大小，hidden_dim是隐层大小，bptt_truncate大家先不用管，我们一会儿会介绍到。

```
class RNNNumpy:

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

```

##前向计算
先来回忆一下前向计算的公式：
st=tanh(U × xt + W × st−1)
ot=softmax(V × st)

这个和卷积神经网络的前向运算完全一样，就是一个根据权重计算下一层/下一个时间点的输出、隐状态的过程。简单的实现如下：

T是x 的长度，也就是句子中的词数，也就是RNN要循环的次数。
我们讲每次循环中的隐藏层结果和输出结果都保存下来，所以先创建固定维度的s,o，并以0填充。
s[-1]也定义一下是因为等等循环第一次t1的时候，需要用到st-1的记忆。

self.U[:, x[t]这个表示的是U×Xt,因为我们之前定义输入xt是one-hot编码的，只有该词对应的索引处为1，其他为0，那么两个相乘其实就是把self.U中的对应词索引的那一列取出来。

所有词都循环计算了一遍后，返回o,s

并且将这个方法加到之前建的RNNNuppy类中。

总结：前向计算，就是根据当前的权重，去从前往后按顺序走一遍神经网络，然后得到一个输出结果。这个输出的预测结果等下就会被拿来和真实的结果做比较。然后选取一个损失函数去衡量他们之间的差值。我们的目标是求得损失函数最小的时候U,V,W的权重。

```
def forward_propagation(self, x):
    # 词数
    T = len(x)
    # 初始化隐藏层结果和输出结果
    s = np.zeros((T+1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
    o = np.zeros((T, self.word_dim))
    # 对句子中的每个词做前向计算
    for t in np.arange(T):
        s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
        o[t] = softmax(self.V.dot(s[t]), [1, vocabulary_size])
    return [o, s]

RNNNumpy.forward_propagation = forward_propagation
```

## 计算损失
正如前文所提，，我们需要一个损失函数/loss function来表征现在的预测结果和真实结果差别有多大，也便于我们之后的迭代和训练。一个比较通用的损失函数叫做互熵损失，如果我们有N个训练样本(这里显然是句子中词的个数咯)，C个类别(这里显然是词表大小咯)，那我们可以这样去定义损失函数描述预测结果o和实际结果y之间的差距:


```
# # 计算总损失，将每个句子中的每个词的损失都加起来
def calculate_total_loss(self, x, y):
    # 初始化损失维0
    l = 0
    # len(y）表示样本中句子的数量
    for i in np.arange(len(y)):
        # 对第i个样本做前向计算得到输出o，o的维度是（该句子中的词数×词表长度）
        o, s = self.forward_propagation(x[i])
        # 我们只关注正确的那个词所在索引的那个概率，所以分别取处每个概率向量中正确词所在的概率
        # 取出来维度是[词数×1]
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        # 将损失叠加到总损失中
        l += -1 * np.sum(np.log(correct_word_predictions))
    return l


# 计算平均损失，将总损失除以样本数量（句子的数量，不是词的数量）
def calculate_loss(self, x, y):
    n = np.sum((len(y_i) for y_i in y))
    return self.calculate_total_loss(x, y) / n

# # 添加到RNNNumpy类中
RNNNumpy.calculate_total_loss = calculate_total_loss
RNNNumpy.calculate_loss = calculate_loss
```

##反向传播
现在损失函数也有咯，我们要做的事情就是最小化这个损失函数，以让我们的预测结果和真实的结果最接近。最常用的优化算法叫做SGD(随机梯度下降)，大家可以理解成我们要从一座山上下山，于是我们每到一个位置，都环顾一下四周，看看最陡(下山最快)的方向是什么，然后顺着它走走，“随机”的意思是我们其实不用全部的样本去计算这个方向，而是用部分样本（有时候是一个）来计算。
计算这个方向就是我们计算梯度的过程，从数学上来说，其实就是在给定损失函数L之后，偏导向量的方向，这里因为有U,V,W三个参数，所以其实我们是要求∂L/∂U,∂L/∂V,∂L/∂W。 

```
def bptt(self, x, y):
    # 词数/循环次数
    T = len(y)
    # 前向计算输出o
    o, s = self.forward_propagation(x)
    # 计算三个参数U V W的梯度,先创建梯度变量，并用0填充
    dl_dU = np.zeros(self.U.shape)
    dl_dV = np.zeros(self.V.shape)
    dl_dW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1
    # 对每一个结果都做反向传播计算
    for t in np.arange(T)[::-1]:
        dl_dV += np.outer(delta_o[t], s[t].T)
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::1]:
            dl_dW += np.outer(delta_t, s[bptt_step-1])
            dl_dU[:, x[bptt_step]] += delta_t
            delta_t = self.W.T.dot(delta_t) * (1 -s[bptt_step - 1] ** 2)
    return [dl_dU, dl_dV, dl_dW]

RNNNumpy.bptt = bptt
```
根据最新的梯度更新参数
```
def sgd_step(self, x, y, learning_rate):
    dl_dU, dl_dV, dl_dW = self.bptt(x, y)
    self.U -= learning_rate * dl_dU
    self.V -= learning_rate * dl_dV
    self.W -= learning_rate * dl_dW

RNNNumpy.sgd_step = sgd_step
```

##开始训练模型
nepoch是对每个训练样本迭代的次数
evaluate_loss_after用来设置每多少次的时候打印一下东东

losses[]这个空的列表等下用来放每次迭代的平均损失，以便我们可以打印出来看损失的变化情况
num_example_seen 是一个计数器，每计来一个样本就记1
```
def train_with_sgd(model, x_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    losses = []
    num_example_seen = 0
    # 迭代
    for epoch in range(nepoch):
        # 有选择地打印损失等信息
        if epoch % evaluate_loss_after == 0:
            # 计算平均损失
            loss = model.calculate_loss(x_train, y_train)
            # 把每次平均损失都追加到losses这个列表中
            losses.append((num_example_seen, loss))
            # 当前时间点
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # 打印时间，迭代第几次，损失
            print "%s:Loss after num_example_seen=%d epoch=%d: %f" % (time, num_example_seen, epoch, loss)
            # 从第二次迭代开始，如果新的损失大于前一次的损失，学习率就减半，并打印新的学习率
            # 损失增大的原因可能是学习率太大，步子迈得太大，一下子迈过了最低点，也是坑爹的。
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate = learning_rate * 0.5
                print "setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        
        # 对与每个样本
        for i in range(len(y_train)):
            # 都通过反向传播去更新参数
            model.sgd_step(x_train[i], y_train[i], learning_rate)
            num_example_seen += 1

```

## 测试
先拿100组样本来做一下测试，循环10次，然后打印每次的损失信息。如果损失是逐渐下降的，说明模型还是有效果的。
```
np.random.seed(10)
model = RNNNumpy(vocabulary_size)
losses = train_with_sgd(model, x_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)

```
来看看打印出来的结果：
```
eading CSV file
Parsed 79170 sentences
Found 65751 unique words tokens.
2016-10-26 08:07:32:Loss after num_example_seen=0 epoch=0: 8.987425
2016-10-26 08:07:44:Loss after num_example_seen=100 epoch=1: 8.985138
2016-10-26 08:07:57:Loss after num_example_seen=200 epoch=2: 8.982309
2016-10-26 08:08:10:Loss after num_example_seen=300 epoch=3: 8.977879
2016-10-26 08:08:23:Loss after num_example_seen=400 epoch=4: 8.966001
2016-10-26 08:08:36:Loss after num_example_seen=500 epoch=5: 6.935774
2016-10-26 08:08:49:Loss after num_example_seen=600 epoch=6: 6.347224
2016-10-26 08:09:02:Loss after num_example_seen=700 epoch=7: 6.091439
2016-10-26 08:09:16:Loss after num_example_seen=800 epoch=8: 5.938676
2016-10-26 08:09:29:Loss after num_example_seen=900 epoch=9: 5.835727
```
我们看到损失在不断下降，说明还是有效果的。但是训练的时间还是挺慢的，100个样本平均13秒才迭代一次，要在CPU上完全训练完79170条训练样本要花很多时间的。。。。

## 预测
如果模型训练好了，我们就可以尝试来让它show一下技能了。

```
def generate_sentence(model):
    # 从 start token/开始符 开始一句话，提取开始付的索引。
    new_sentence = [word_to_index[sentence_start_token]]
    # 直到拿到一个 end token/结束符
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

num_sentences = 10
senten_min_length = 7

for i in range(num_sentences):
    sent = []
    # 短句就不要咯，留下长的就好
    while len(sent) > senten_min_length:
        sent = generate_sentence(model)
    print " ".join(sent)
```


  [1]: http://static.zybuluo.com/wangcao/8vn1gof8lg7j1lva3ski2z1x/2016-10-25%2017-59-36%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png