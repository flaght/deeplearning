# Deep Residual Network

标签（空格分隔）： 王小草深度学习笔记

---

##1. 背景
原论文链接：https://arxiv.org/abs/1512.03385

当神经网络深度很深的时候，残差做后向计算求导时会变得非常非常小，所以训练起来非常困难。但是往往神经网络越深它的效果也越好。

所以，得想一个办法得让神经网络又深又好训练。于是微软亚洲研究院提出ResNet，在2015年的ILSVRC比赛中获得了冠军，比VGG还要深8倍。（152层）

##2. 结构
深层的CNN训练困难在于梯度衰减，即使使用bantch normalization,几十层的CNN也非常难训练。
离输入层越远，残差传回来的信号也就非常弱了，从而导致了失真。

于是ResNet涉及了这样一个结构：
![QQ截图20161114151147.png-40kB][1]

F(x)是一个残差的隐射，将它加上从根部传过来的原始的信息x再去经过激励函数relu往下走。

整个ResNet的结构可以如下图：
![QQ截图20161114152005.png-74.7kB][2]

##3.Deep Residual Network tflearn实现

```
from __future__ import division, print_function, absolute_import

import tflearn

# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 5

# Data loading
from tflearn.datasets import cifar10
(X, Y), (testX, testY) = cifar10.load_data()
Y = tflearn.data_utils.to_categorical(Y, 10)
testY = tflearn.data_utils.to_categorical(testY, 10)

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([32, 32], padding=4)

# Building Residual Network
net = tflearn.input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.residual_block(net, n, 16)
net = tflearn.residual_block(net, 1, 32, downsample=True)
net = tflearn.residual_block(net, n-1, 32)
net = tflearn.residual_block(net, 1, 64, downsample=True)
net = tflearn.residual_block(net, n-1, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 10, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_cifar10',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

model.fit(X, Y, n_epoch=200, validation_set=(testX, testY),
          snapshot_epoch=False, snapshot_step=500,
          show_metric=True, batch_size=128, shuffle=True,
          run_id='resnet_cifar10')
```


  [1]: http://static.zybuluo.com/wangcao/ejzbvw43pw8ikgo66uuvzyve/QQ%E6%88%AA%E5%9B%BE20161114151147.png
  [2]: http://static.zybuluo.com/wangcao/ftzsbwj0hg87ciidjj6eix3s/QQ%E6%88%AA%E5%9B%BE20161114152005.png