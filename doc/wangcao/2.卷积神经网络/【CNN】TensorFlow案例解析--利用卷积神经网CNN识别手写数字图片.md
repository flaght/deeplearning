# 【CNN】TensorFlow案例解析--利用卷积神经网CNN识别手写数字图片

标签（空格分隔）： 王小草深度学习笔记

---
## 写在前面
TensorFlow是谷歌2015年开源的一款人工智能框架，官方发布出来后，极客学院wiki团队群策群力马上将英文版的官方文件都翻译成了中文。github上搜索“tensorflow",第一条是英文版官方(Python版的），第二条就是中文官方。（前些日子出了R语言版的官方文档，第三条才是中文python)。另外，也可以直接百度搜索“tensorflow中文社区”。（但是发布出来之后谷歌官方是不断地在更新版本和代码的，中文翻译好像没跟上，所以建议还是直接看英文官方版的才不会太被坑）

Tensorflow其实没有宣传中的那么酷炫，深度学习的框架其实就是将各种类型的神经网络分装成一个接口，我们使用的时候直接调用这些接口，就能完成模型的创建与训练。

关于tensorflow的基础，包括安装，使用等，请见github上的官方文档，都有详细介绍，一看就明白的。

本文要讲解的是一个官方发布的小案例--利用卷积神经网CNN识别手写数字图片。尽管官方文档中也仔细讲述了代码的基本含义，但对于初次接触卷积神经网络的童鞋来说，可能理解起来仍然可远观不可亵玩。本文给每条代码都加上了注释，并且在最后展示出运行结果供大家深入理解。

当然了，如果对神经网络与卷积神经网络的知识不曾了解的小伙伴，直接阅读代码可能不知所云，推荐大家阅读笔者的另一批笔记--“王小草深度学习笔记”，里面循序渐进地讲述了深度学习的知识，从最基础的神经网络到卷积到word2vector再到RNN与迁移学习等等。理解理论之后再来阅读卷积神经网络的tensorflow实现代码，虽然写得不好但也能帮助理解。所有文档都会不定期发布在我的公众微信号（王小草之大数据人工智能）与CSDN博客（用户名为‘点绛唇--王小草）上。

##案例介绍
之前也说过CNN做图像处理的效果挺好的。今天这个案例使用的数据是MNIST，MNIST是一个入门级的计算机视觉数据集，它的数据是各种手写数字的图片。每一张图片对应一个标签，告诉我们这个图片上的数字是几。

我们要做的是对这些图片数据建立模型，并且进行训练和学习，使得模型对其他未知的手写字图片能够自动地精确地识别上面的数字。

对图像做分类的算法非常多，可以用机器学习的算法，比如softmax regression,或者是深度学习的算法，比如深层神经网络DNN，更好的还有是卷积神经网络CNN。
本文给出CNN的代码与解释。

## 程序实现

```
#!/usr/bin/python
# -*- coding:utf-8 -*-

from tensorflow.examples.tutorials.mnist.input_data import *
import tensorflow as tf

# 读取数据集，read_data_sets是一个已经封装好的方法，会去直接下载数据并且做预处理
mnist = read_data_sets("MNIST_data/", one_hot=True)

# # 定义数据，设置占位符
# 设置特征与标签的占位符，特征集是n×784维，标签集维n×10维，n是可以调节的
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder("float", [None, 10])
# 设置dropout的占位符，dropout用于防止过拟合
keep_pro = tf.placeholder("float")
# 将平铺的特征重构成28×28的图片像素维度，因为使用的是黑白图片，所以颜色通道维1,因为要取出所有数据，所以索引维-1
x_image = tf.reshape(x, [-1, 28, 28, 1])


# # 定义函数以方便构造网络
# 初始化权重,传入大小参数，truncated_normal函数使得w呈正太分布，
# stddev设置标准差为0.1。也就是说输入形状大小，输出正太分布的随机参数作为权重变量矩阵
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
    
# 初始化偏执项，传入矩阵大小的参数，生成该大小的值全部为0.1的矩阵
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷基层，步长为1,周围补0，输入与输出的数据大小一样（可得到补全的圈数）
def conv2d(a, w):
    return tf.nn.conv2d(a, w, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化层,kernel大小为2,步长为2,周围补0，输入与输出的数据大小一样（可得到补全的圈数）
def max_pool_2x2(a):
    return tf.nn.max_pool(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# # 进入卷积层与池化层1
# 初始化权重，传入权重大小，窗口的大小是5×5,所以指向每个卷积层的权重也是5×5,
# 卷积层的神经元的个数是32,总共只有1个面（1个颜色通道）
w_conv1 = weight_variable([5, 5, 1, 32])
# 32个神经元就需要32个偏执项
b_conv1 = bias_variable([32])
# 将卷积层相对应的数据求内积再加上偏执项的这个线性函数，放入激励层relu中做非线性打转换，输出的大小是28×28×32
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
# 将卷积层输出的数据传入池化层，根据函的设定，窗口大小维2×2,步长为2,输出的大小就降到来14×14×32
h_pool1 = max_pool_2x2(h_conv1)

# # 进入卷积层与池化层2
# 第2层卷积层由64个神经元，1个神经元要初始化的权重维度是5×5×32
w_conv2 = weight_variable([5, 5, 32, 64])
# 偏执项的数目和神经元的数目一样
b_conv2 = bias_variable([64])
# 将池化层1的输出与卷积层2的权重做内积再加上偏执项，然后进入激励函数relu，输出维度为14×14×64
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
# 进入池化层，输出减半为7×7×64
h_pool2 = max_pool_2x2(h_conv2)
# # 进入全连接层1
# 初始化全链接层的权重，全了链接层有1024个神经元，每个都与池化层2的输出数据全部连接
w_fc1 = weight_variable([7*7*64, 1024])
# 偏执项也等于神经元的个数1024
b_fc1 = bias_variable([1024])
# 将池化层的输出数据拉平为1行7×7×64列打矩阵，-1表示把所有都拿出来
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# 全连接计算，线性运算后再输入激励函数中，最后输出1024个数据
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
# 使用dropout防止过拟合
h_fc1_drop = tf.nn.dropout(h_fc1, keep_pro)

# # 进入全连接层2
# 初始化权重，全连接层2有10个神经元，上一层打输入是1024
w_fc2 = weight_variable([1024, 10])
# 偏执项为10
b_fc2 = bias_variable([10])
# 全连接的计算，然后再过一个softmax函数，输出为10个数据（10个概率）
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# # 损失函数最小的最优化计算
# 交叉熵作为目标函数计算
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# 目标函数最小训练模型，估计参数，使用的是ADAM优化器来做梯度下降
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 计算预测正确的个数，tf.argmax是寻找一个tensor中每个维度的最大值所在的索引
# 因为类别是用0，1表示的，所以找出1所在打索引就能找到数字打类别
# tf.equals是检测预测与真实的标签是否一致，返回的是布尔值，true,false
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# 计算正确率,用tf.cast来将true,false转换成1,0,然后计算正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# # 创建会话,初始化变量
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# # 执行循环
for i in range(2000):
    # 每批取出50个训练样本
    batch = mnist.train.next_batch(50)
    # 循环次数是100的倍数的时候，打印东东
    if i % 100 == 0:
        # 计算正确率，
        train_accuracy = accuracy.eval(feed_dict={
           x: batch[0], y_: batch[1], keep_pro: 1.0})
        # 打印
        print "step %d, training accuracy %g" % (i, train_accuracy)
    # 执行训练模型
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_pro: 0.5})
# 打印测试集正确率
print "test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_pro: 1.0
    })

```

这个代码设置了循环2000次，每循环100次的时候，会打印出训练集的误差。打印结果如下：
```
step 0, training accuracy 0.1
step 100, training accuracy 0.84
step 200, training accuracy 0.9
step 300, training accuracy 0.86
step 400, training accuracy 0.98
step 500, training accuracy 0.94
step 600, training accuracy 0.98
step 700, training accuracy 0.96
step 800, training accuracy 0.9
step 900, training accuracy 1
step 1000, training accuracy 0.94
step 1100, training accuracy 0.94
step 1200, training accuracy 0.98
step 1300, training accuracy 1
step 1400, training accuracy 0.98
step 1500, training accuracy 0.94
step 1600, training accuracy 0.94
step 1700, training accuracy 0.94
step 1800, training accuracy 0.96
step 1900, training accuracy 0.98
```

在最后，会打印出测试集的正确率：
```
test accuracy 0.9763
```
可以发现用卷积神经网络做这次分类，正确率其实是挺高的，在0.976左右，我这边设置的循环次数是2000，在官网上设置的是20000，正确率可以高高达0.99+，是不是很牛逼哄哄呢~

其实我觉得要用tensorflow实现神经网络的运算也许不难，难处应是在对卷积神经网络中各类型的层级的运作理解，这样才会知道每层的计算需要多少维度的权重，每层的输入输出是什么样的数据，理解整个卷积神经网络的每一步结构才能真正理解整个程序的运作，也才能灵活地使用tensorflow提供的各自函数与平台。下面这幅图是我在某个博客上截取的，描述的就是本文案例的整个过程，非常有助于理解，在此分享给大家。
![QQ截图20160929181044.png-300.3kB][1]


  [1]: http://static.zybuluo.com/wangcao/sj0acpyxfrudy6cysuhewgv6/QQ%E6%88%AA%E5%9B%BE20160929181044.png