# Lenet And Alexnet

标签（空格分隔）： 王小草深度学习笔记

---

时间：2016年11月14日
笔记整理者：王小草
笔记声明：本笔记是整合了多个博客中优秀的内容与图片，并非原创
笔记来源：http://m.blog.csdn.net/article/details?id=51440344
http://blog.csdn.net/sunbaigui/article/details/39938097
http://hacker.duanshishi.com/?p=1661

人神共愤的周一又来。
这周主要想将一些优秀的CNN的框架的结构都整理一遍，包括Lenet, Alexnet, ZFnet,VGG, GoogleLnet, ResNet.过目一下这几个模型的背景：
![QQ截图20161114140832.png-122.7kB][1]

##CNN的演化
下图是截取刘昕博士总结的CNN演化历史图。起点是神经认知机模型，此时已经出现了卷积结构，经典的LeNet诞生于1998年。然而之后CNN的锋芒开始被SVM等手工设计的特征盖过。随着ReLU和dropout的提出，以及GPU和大数据带来的历史机遇，CNN在2012年迎来了历史突破–AlexNet.
![QQ截图20161114113400.png-63.3kB][2]

本文要介绍的是Lenet和Alexnet的框架与结构。

##1.Lenet
下图是lenet的结构，与目前不断改进中的CNN一样，也同样具有卷积层，池化层，全链接层。
![QQ截图20161114113734.png-84kB][3]

###1.1 结构
**输入层**
输入层是一张黑白图片，大小是32*32（1个颜色通道）。

**卷积层1--C1**
输入层的下一层便是第1个卷积层了。这个卷积层有6个卷积核（每个神经元对应一个卷积核），核大小是5*5.即输出的是6个28*28的feature map,每个feature map都提取了一个局部特征。

这一层需要训练的参数个数是（5*5+1）*6 = 156 。加1是因为每一个卷积核线性计算中还有一个bais偏执项参数。

如此一来，C1中的链接数共有156*（28*28） = 122304个。

**池化层1--S2**
池化层也叫向下采样层，是为了降低网络训练参数及模型的过拟合程度。
C1的输出就是S2的输入，S2的输入大小为6 * 28 * 28.

窗口大小为2*2，步长为2，所以采样之后的S2输出是6 * 14 * 14

**卷积层2--C3**
C3有16个卷积核（16个神经元），同样也是通过5 * 5的卷积核，得到16个10 * 10 的feature map。

在C1中是有6个卷积核的，现在有16个，每个卷积核都是提取了一个局部的特征。所以16个卷积核其实代表着C1中6中特征的组合。

**池化层2--S4**
同样，通过对C3的向下采样，输出之后的大小为16 * 5 * 5.

**卷积层3--C5**
C5有120个卷积核，每个单元与S4的全部16个单元的5*5邻域相连，由于S4层特征图的大小也为5*5（同滤波器一样），故C5特征图的大小为1*1：这构成了S4和C5之间的全连接。之所以仍将C5标示为卷积层而非全相联层，是因为如果LeNet-5的输入变大，而其他的保持不变，那么此时特征图的维数就会比1*1大。C5层有48120个可训练连接。

**全链接层1--F6**
F6层有84个单元（之所以选这个数字的原因来自于输出层的设计），与C5层全相连。有10164个可训练参数。如同经典神经网络，F6层计算输入向量和权重向量之间的点积，再加上一个偏置。然后将其传递给sigmoid函数产生单元i的一个状态。

**输出层--output**
输出层由欧式径向基函数（Euclidean Radial Basis Function）单元组成，每类一个单元，每个有84个输入。 
换句话说，每个输出RBF单元计算输入向量和参数向量之间的欧式距离。输入离参数向量越远，RBF输出的越大。用概率术语来说，RBF输出可以被理解为F6层配置空间的高斯分布的负log-likelihood。给定一个输式，损失函数应能使得F6的配置与RBF参数向量（即模式的期望分类）足够接近。

##2.Alexnet
AlexNet 可以说是具有历史意义的一个网络结构，可以说在AlexNet之前，深度学习已经沉寂了很久。历史的转折在2012年到来，AlexNet 在当年的ImageNet图像分类竞赛中，top-5错误率比上一年的冠军下降了十个百分点，而且远远超过当年的第二名。

AlexNet 之所以能够成功，深度学习之所以能够重回历史舞台，原因在于：
(1)非线性激活函数：ReLU
(2)防止过拟合的方法：Dropout，Data augmentation
(3)大数据训练：百万级ImageNet图像数据
(4)其他：GPU实现，LRN归一化层的使用

###2.1 结构
Alexnet的结构如下：
![QQ截图20161114132145.png-47.4kB][4]

alexnet总共包括8层，其中前5层convolutional，后面3层是full-connected，文章里面说的是减少任何一个卷积结果会变得很差，下面我来具体讲讲每一层的构成：

第一层卷积层 输入图像为227*227*3(paper上貌似有点问题224*224*3)的图像，使用了96个kernels（96,11,11,3），以4个pixel为一个单位来右移或者下移，能够产生5555个卷积后的矩形框值，然后进行response-normalized（其实是Local Response Normalized，后面我会讲下这里）和pooled之后，pool这一层好像caffe里面的alexnet和paper里面不太一样，alexnet里面采样了两个GPU，所以从图上面看第一层卷积层厚度有两部分，池化pool_size=(3,3),滑动步长为2个pixels，得到96个2727个feature。
第二层卷积层使用256个（同样，分布在两个GPU上，每个128kernels（5*5*48）），做pad_size(2,2)的处理，以1个pixel为单位移动（感谢网友指出），能够产生27*27个卷积后的矩阵框，做LRN处理，然后pooled，池化以3*3矩形框，2个pixel为步长，得到256个13*13个features。
第三层、第四层都没有LRN和pool，第五层只有pool，其中第三层使用384个kernels（3*3*384，pad_size=(1,1),得到384*15*15，kernel_size为（3，3),以1个pixel为步长，得到384*13*13）；第四层使用384个kernels（pad_size(1,1)得到384*15*15，核大小为（3，3）步长为1个pixel，得到384*13*13）；第五层使用256个kernels（pad_size(1,1)得到384*15*15，kernel_size(3,3)，得到256*13*13，pool_size(3，3）步长2个pixels，得到256*6*6）。
全连接层： 前两层分别有4096个神经元，最后输出softmax为1000个（ImageNet），注意caffe图中全连接层中有relu、dropout、innerProduct。

Alexnet总共有5个卷积层，具体的数据流如下：
![QQ截图20161114132850.png-185.7kB][5]

如下有更清晰的数据结构（截图自http://blog.csdn.net/sunbaigui/article/details/39938097）
![QQ截图20161114133243.png-78.5kB][6]
![QQ截图20161114133111.png-80.1kB][7]
![QQ截图20161114133123.png-91.8kB][8]
![QQ截图20161114133137.png-115kB][9]
![QQ截图20161114133149.png-82.9kB][10]

###2.2 优点
**Data augmentation**
有一种观点认为神经网络是靠数据喂出来的，若增加训练数据，则能够提升算法的准确率，因为这样可以避免过拟合，而避免了过拟合你就可以增大你的网络结构了。当训练数据有限的时候，可以通过一些变换来从已有的训练数据集中生成一些新的数据，来扩大训练数据的size。
其中，最简单、通用的图像数据变形的方式:
从原始图像（256,256）中，随机的crop出一些图像（224,224）。【平移变换，crop】
水平翻转图像。【反射变换，flip】
给图像增加一些随机的光照。【光照、彩色变换，color jittering】

AlexNet 训练的时候，在data augmentation上处理的很好：
随机crop。训练时候，对于256＊256的图片进行随机crop到224＊224，然后允许水平翻转，那么相当与将样本倍增到((256-224)^2)*2=2048。
测试时候，对左上、右上、左下、右下、中间做了5次crop，然后翻转，共10个crop，之后对结果求平均。作者说，不做随机crop，大网络基本都过拟合(under substantial overfitting)。
对RGB空间做PCA，然后对主成分做一个(0, 0.1)的高斯扰动。结果让错误率又下降了1%。

**ReLU 激活函数**
Sigmoid 是常用的非线性的激活函数，它能够把输入的连续实值“压缩”到0和1之间。特别的，如果是非常大的负数，那么输出就是0；如果是非常大的正数，输出就是1. 
但是它有一些致命的 缺点：
Sigmoids saturate and kill gradients. sigmoid 有一个非常致命的缺点，当输入非常大或者非常小的时候，会有饱和现象，这些神经元的梯度是接近于0的。如果你的初始值很大的话，梯度在反向传播的时候因为需要乘上一个sigmoid 的导数，所以会使得梯度越来越小，这会导致网络变的很难学习。
Sigmoid 的 output 不是0均值. 这是不可取的，因为这会导致后一层的神经元将得到上一层输出的非0均值的信号作为输入。 
产生的一个结果就是：如果数据进入神经元的时候是正的(e.g. x>0 elementwise in f=wTx+b)，那么 w 计算出的梯度也会始终都是正的。 
当然了，如果你是按batch去训练，那么那个batch可能得到不同的信号，所以这个问题还是可以缓解一下的。因此，非0均值这个问题虽然会产生一些不好的影响，不过跟上面提到的 kill gradients 问题相比还是要好很多的。

Alex用ReLU代替了Sigmoid，发现使用 ReLU 得到的SGD的收敛速度会比 sigmoid/tanh 快很多。
主要是因为它是linear，而且 non-saturating（因为ReLU的导数始终是1），相比于 sigmoid/tanh，ReLU 只需要一个阈值就可以得到激活值，而不用去算一大堆复杂的运算。

**Dropout**
结合预先训练好的许多不同模型，来进行预测是一种非常成功的减少测试误差的方式（Ensemble）。但因为每个模型的训练都需要花了好几天时间，因此这种做法对于大型神经网络来说太过昂贵。
然而，AlexNet 提出了一个非常有效的模型组合版本，它在训练中只需要花费两倍于单模型的时间。这种技术叫做Dropout，它做的就是以0.5的概率，将每个隐层神经元的输出设置为零。以这种方式“dropped out”的神经元既不参与前向传播，也不参与反向传播。
所以每次输入一个样本，就相当于该神经网络就尝试了一个新的结构，但是所有这些结构之间共享权重。因为神经元不能依赖于其他特定神经元而存在，所以这种技术降低了神经元复杂的互适应关系。
正因如此，网络需要被迫学习更为鲁棒的特征，这些特征在结合其他神经元的一些不同随机子集时有用。在测试时，我们将所有神经元的输出都仅仅只乘以0.5，对于获取指数级dropout网络产生的预测分布的几何平均值，这是一个合理的近似方法。

**多GPU训练**
单个GTX 580 GPU只有3GB内存，这限制了在其上训练的网络的最大规模。因此他们将网络分布在两个GPU上。 
目前的GPU特别适合跨GPU并行化，因为它们能够直接从另一个GPU的内存中读出和写入，不需要通过主机内存。
他们采用的并行方案是：在每个GPU中放置一半核（或神经元），还有一个额外的技巧：GPU间的通讯只在某些层进行。

![QQ截图20161114133650.png-47.8kB][11]

例如，第3层的核需要从第2层中所有核映射输入。然而，第4层的核只需要从第3层中位于同一GPU的那些核映射输入。
Local Responce Normalization
一句话概括：本质上，这个层也是为了防止激活函数的饱和的。

###2.3 AlexNet On Tensorflow
代码位置：
http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

tflearn里面有一个alexnet来分类Oxford的例子，好开心，在基于tflearn对一些日常layer的封装，代码量只有不到50行，看了下内部layer的实现，挺不错的，写代码的时候可以多参考参考，代码地址https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py.
```
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.oxflower17 as oxflower17
X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_oxflowers17')
```


  [1]: http://static.zybuluo.com/wangcao/3e5bgj90p5mnp5f7dnheszp9/QQ%E6%88%AA%E5%9B%BE20161114140832.png
  [2]: http://static.zybuluo.com/wangcao/7mxxxha07yxg7o5xevrezkiu/QQ%E6%88%AA%E5%9B%BE20161114113400.png
  [3]: http://static.zybuluo.com/wangcao/2lwgfi0q3z0dfcgo0rs6dree/QQ%E6%88%AA%E5%9B%BE20161114113734.png
  [4]: http://static.zybuluo.com/wangcao/mh9ed3frhlsvaa1o3ta6lnra/QQ%E6%88%AA%E5%9B%BE20161114132145.png
  [5]: http://static.zybuluo.com/wangcao/41tsz3d8lurr8pb6262zvywe/QQ%E6%88%AA%E5%9B%BE20161114132850.png
  [6]: http://static.zybuluo.com/wangcao/5gbfvmzrmwsq7imhpaongnb3/QQ%E6%88%AA%E5%9B%BE20161114133243.png
  [7]: http://static.zybuluo.com/wangcao/rrajrmjohv58gd6m8pzo9cpf/QQ%E6%88%AA%E5%9B%BE20161114133111.png
  [8]: http://static.zybuluo.com/wangcao/o2hrs82htobtknzdxlvbdjfa/QQ%E6%88%AA%E5%9B%BE20161114133123.png
  [9]: http://static.zybuluo.com/wangcao/ourpp2xoztg54x81r5ggstta/QQ%E6%88%AA%E5%9B%BE20161114133137.png
  [10]: http://static.zybuluo.com/wangcao/wmaj94u851ot7hmqmo60z4zd/QQ%E6%88%AA%E5%9B%BE20161114133149.png
  [11]: http://static.zybuluo.com/wangcao/usliib567t3acdnzv59dty44/QQ%E6%88%AA%E5%9B%BE20161114133650.png