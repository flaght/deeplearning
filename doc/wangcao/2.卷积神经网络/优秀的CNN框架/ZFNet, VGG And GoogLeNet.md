# ZFNet, VGG And GoogLeNet

标签（空格分隔）： 王小草深度学习笔记

---

##1. ZFNet
首先来简单地介绍一下ZFNet这个CNN模型。它是2013年ILSVRC比赛的冠军。它是在Alexnet的基础上做一点点的改进和变换。我结构如下：

![QQ截图20161114141203.png-142kB][1]

ZFNet与Alexnet有两点差别
1.在卷积层1中，VGG的卷积核是11*11大小，并且步长为4；在ZFNet中，卷积核大小改为了7 * 7， 步长改为了2.
2.在卷积层3,4,5中，神经元的个数分别由VGG的384,384,256变成了512,1024,512.

其余结构与AlexNet完全一致。

ZFNet将top5的错误率从15.4%下降到了14.8%，提升并没有特别惊人。

##2.VGG
###2.1 结构
VGGnet是Oxford的Visual Geometry Group的team，在ILSVRC 2014上的相关工作，主要工作是证明了增加网络的深度能够在一定程度上影响网络最终的性能，如下图，文章通过逐步增加网络深度来提高性能，虽然看起来有一点小暴力，没有特别多取巧的，但是确实有效，很多pretrained的方法就是使用VGG的model（主要是16和19），VGG相对其他的方法，参数空间很大，最终的model有500多m，alnext只有200m，googlenet更少，所以train一个vgg模型通常要花费更长的时间，所幸有公开的pretrained model让我们很方便的使用。paper中的几种模型如下：

![QQ截图20161114141739.png-315.7kB][2]

![QQ截图20161114141706.png-36.7kB][3]

论文的作者试验了多种深度与结构的CNN， 在16层的时候达到了最好。所以我们现在一般都使用VGG-16，VGG-19.

以下是VGG每一层的数据维度
![QQ截图20161114142056.png-85.6kB][4]

以下是计算VGG每一层需要的显存，是可以通过参数的数量进行估算的。要运行一个VGG,一张图片需要给它93M，而且这只是前向计算的，如果还药后向计算，那么要93*2.
![QQ截图20161114142128.png-516kB][5]

### 2.2 VGG on tensorflow

```
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# Data loading and preprocessing
import tflearn.datasets.oxflower17 as oxflower17
X, Y = oxflower17.load_data(one_hot=True)

# Building 'VGG Network'
network = input_data(shape=[None, 224, 224, 3])

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')

network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=500, shuffle=True,
          show_metric=True, batch_size=32, snapshot_step=500,
          snapshot_epoch=False, run_id='vgg_oxflowers17')
```

## GoogLeNet
参考博客“http://hacker.duanshishi.com/?p=1678

GoogLeNet是ILSVRC 2014的冠军，主要是致敬经典的LeNet-5算法，主要是Google的team成员完成，paper见Going Deeper with Convolutions.相关工作主要包括LeNet-5、Gabor filters、Network-in-Network.Network-in-Network改进了传统的CNN网络，采用少量的参数就轻松地击败了AlexNet网络，使用Network-in-Network的模型最后大小约为29MNetwork-in-Network caffe model.GoogLeNet借鉴了Network-in-Network的思想

![QQ截图20161114144415.png-170.6kB][6]

**总体结构：** 

1.包括Inception模块的所有卷积，都用了修正线性单元（ReLU）； 

2.网络的感受野大小是224x224，采用RGB彩色通道，且减去均值； 

3.#3x3 reduce和#5x5 reduce分别表示3x3和5x5的卷积前缩减层中1x1滤波器的个数；pool proj表示嵌入的max-pooling之后的投影层中1x1滤波器的个数；缩减层和投影层都要用ReLU； 

4.网络包含22个带参数的层（如果考虑pooling层就是27层），独立成块的层总共有约有100个； 

5.网络中间的层次生成的特征会非常有区分性，给这些层增加一些辅助分类器。这些分类器以小卷积网络的形式放在Inception(4a)和Inception(4b)的输出上。在训练过程中，损失会根据折扣后的权重（折扣权重为0.3）叠加到总损失中。

**辅助分类器的具体细节：** 

1.均值pooling层滤波器大小为5x5，步长为3，(4a)的输出为4x4x512，(4d)的输出为4x4x528； 

2.1x1的卷积有用于降维的128个滤波器和修正线性激活； 

3.全连接层有1024个单元和修正线性激活； 

4.dropout层的dropped的输出比率为70%； 

5.线性层将softmax损失作为分类器（和主分类器一样预测1000个类，但在inference时移除）。


  [1]: http://static.zybuluo.com/wangcao/oy6q3wvfmhobbpb51dt5yli6/QQ%E6%88%AA%E5%9B%BE20161114141203.png
  [2]: http://static.zybuluo.com/wangcao/9en82bgtnakwmpdbuitogi4k/QQ%E6%88%AA%E5%9B%BE20161114141739.png
  [3]: http://static.zybuluo.com/wangcao/j09jr7aqxo8p4u1i2dfvrbn0/QQ%E6%88%AA%E5%9B%BE20161114141706.png
  [4]: http://static.zybuluo.com/wangcao/pqz8lny1umgqqjz72fitxx5m/QQ%E6%88%AA%E5%9B%BE20161114142056.png
  [5]: http://static.zybuluo.com/wangcao/0owm83ogic9cjl7n1nwpnpwv/QQ%E6%88%AA%E5%9B%BE20161114142128.png
  [6]: http://static.zybuluo.com/wangcao/4rk4grb1x5lldmdpq01z1tmv/QQ%E6%88%AA%E5%9B%BE20161114144415.png