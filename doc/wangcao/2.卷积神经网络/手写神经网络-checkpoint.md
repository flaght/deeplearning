
# 手写神经网络
这个ipython notebook是手写的多层神经网络(都是全连接层)，然后在CIFAR-10数据集上做实验<br>
[@寒小阳](http://blog.csdn.net/han_xiaoyang)<br>
2016年5月


```python
# 初始设定，可以略过

import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2

def rel_error(x, y):
  """ 返回相对误差 """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
```

我们手写的神经网络非常简单，模型其实就是最后的权重，我们存在一个python dict里面，按层存了W和偏移项b<br>
先练练手，我们初始化一个给定初始权重的神经网络，以及一部分数据。


```python
# 随机初始化一个试验模型(其实就是存在dic中的权重)和数据集
input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
  model = {}
  model['W1'] = np.linspace(-0.2, 0.6, num=input_size*hidden_size).reshape(input_size, hidden_size)
  model['b1'] = np.linspace(-0.3, 0.7, num=hidden_size)
  model['W2'] = np.linspace(-0.4, 0.1, num=hidden_size*num_classes).reshape(hidden_size, num_classes)
  model['b2'] = np.linspace(-0.5, 0.9, num=num_classes)
  return model

def init_toy_data():
  X = np.linspace(-0.2, 0.5, num=num_inputs*input_size).reshape(num_inputs, input_size)
  y = np.array([0, 1, 2, 2, 1])
  return X, y

model = init_toy_model()
X, y = init_toy_data()
```

# 前向计算: 获取得分
这个部分有点像我们前面写的linear SVM和Softmax分类器：其实做的事情都一样，我们根据数据和权重去计算每个类的得分，损失函数值，以及参数上的梯度。


```python
from nn.classifiers.neural_net import two_layer_net

scores = two_layer_net(X, model, verbose=True)
print scores
correct_scores = [[-0.5328368, 0.20031504, 0.93346689],
 [-0.59412164, 0.15498488, 0.9040914 ],
 [-0.67658362, 0.08978957, 0.85616275],
 [-0.77092643, 0.01339997, 0.79772637],
 [-0.89110401, -0.08754544, 0.71601312]]

# 我们前向运算计算得到的得分和实际的得分应该差别很小才对
print '前向运算得到的得分和实际的得分差别:'
print np.sum(np.abs(scores - correct_scores))
```

    Layer 1 result shape: (5, 10)
    Layer 2 result shape: (5, 3)
    [[-0.5328368   0.20031504  0.93346689]
     [-0.59412164  0.15498488  0.9040914 ]
     [-0.67658362  0.08978957  0.85616275]
     [-0.77092643  0.01339997  0.79772637]
     [-0.89110401 -0.08754544  0.71601312]]
    前向运算得到的得分和实际的得分差别:
    3.84868228918e-08
    

# 前向运算：计算损失
这里的loss包括数据损失和正则化损失


```python
reg = 0.1
loss, _ = two_layer_net(X, model, y, reg)
correct_loss = 1.38191946092

# 应该差值是很小的
print '我们计算到的损失和真实的损失值之间差别:'
print np.sum(np.abs(loss - correct_loss))
```

    我们计算到的损失和真实的损失值之间差别:
    4.67692551354e-12
    

# 反向传播部分
咱们得计算loss在`W1`, `b1`, `W2`和`b2`上的梯度，就是反向传播的实现，不过注意梯度计算的时候要进行梯度检验哦:


```python
from nn.gradient_check import eval_numerical_gradient

# 使用数值梯度去检查反向传播的计算

loss, grads = two_layer_net(X, model, y, reg)

# 各参数应该比 1e-8 要小才保险
for param_name in grads:
  param_grad_num = eval_numerical_gradient(lambda W: two_layer_net(X, model, y, reg)[0], model[param_name], verbose=False)
  print '%s 最大相对误差: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))
  
```

    W1 最大相对误差: 4.426512e-09
    W2 最大相对误差: 8.023740e-10
    b2 最大相对误差: 8.190173e-11
    b1 最大相对误差: 5.435432e-08
    

# 训练神经网络
用定步长SGD和SGD with Momentum完成最小化损失函数。<br>
具体的实现在`classifier_trainer.py`文件的`ClassifierTrainer`类里。<br>
先试试定步长的SGD


```python
from nn.classifier_trainer import ClassifierTrainer

model = init_toy_model()
trainer = ClassifierTrainer()
# 这个地方是自己手造的数据，量不大，所以其实sample_batches就设为False了，直接全量梯度下降
best_model, loss_history, _, _ = trainer.train(X, y, X, y,
                                             model, two_layer_net,
                                             reg=0.001,
                                             learning_rate=1e-1, momentum=0.0, learning_rate_decay=1,
                                             update='sgd', sample_batches=False,
                                             num_epochs=100,
                                             verbose=False)
print 'Final loss with vanilla SGD: %f' % (loss_history[-1], )
```

    starting iteration  0
    Final loss with vanilla SGD: 0.940686
    

下面是使用**momentum update**的步长更新策略的SGD, 你会看到最后的loss值会比上面要小一些


```python
model = init_toy_model()
trainer = ClassifierTrainer()
# call the trainer to optimize the loss
# Notice that we're using sample_batches=False, so we're performing Gradient Descent (no sampled batches of data)
best_model, loss_history, _, _ = trainer.train(X, y, X, y,
                                             model, two_layer_net,
                                             reg=0.001,
                                             learning_rate=1e-1, momentum=0.9, learning_rate_decay=1,
                                             update='momentum', sample_batches=False,
                                             num_epochs=100,
                                             verbose=False)
correct_loss = 0.494394
print 'Final loss with momentum SGD: %f. We get: %f' % (loss_history[-1], correct_loss)
```

    starting iteration  0
    Final loss with momentum SGD: 0.494394. We get: 0.494394
    

当然也可以试试课上提到的 **RMSProp** 方式做SGD最优化:


```python
model = init_toy_model()
trainer = ClassifierTrainer()
# call the trainer to optimize the loss
# Notice that we're using sample_batches=False, so we're performing Gradient Descent (no sampled batches of data)
best_model, loss_history, _, _ = trainer.train(X, y, X, y,
                                             model, two_layer_net,
                                             reg=0.001,
                                             learning_rate=1e-1, momentum=0.9, learning_rate_decay=1,
                                             update='rmsprop', sample_batches=False,
                                             num_epochs=100,
                                             verbose=False)
correct_loss = 0.439368
print 'Final loss with RMSProp: %f. We get: %f' % (loss_history[-1], correct_loss)
```

    starting iteration  0
    Final loss with RMSProp: 0.439368. We get: 0.439368
    

# 载入数据
我们手写了一个2层的全连接神经网络（感知器），并在 CIFAR-10数据集上试试效果。


```python
from nn.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    载入CIFAR-10数据集，并做预处理。这一步和前一节课用softmax和SVM分类是一样的
    """
    cifar10_dir = 'nn/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # 采样数据
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # 去均值
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # 调整维度
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# 看看数据维度
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape
```

    Train data shape:  (49000, 3072)
    Train labels shape:  (49000,)
    Validation data shape:  (1000, 3072)
    Validation labels shape:  (1000,)
    Test data shape:  (1000, 3072)
    Test labels shape:  (1000,)
    

# 训练神经网络
我们使用SGD with momentum进行最优化。每一轮迭代以后，我们把学习率衰减一点点。


```python
from nn.classifiers.neural_net import init_two_layer_model
from nn.classifier_trainer import ClassifierTrainer

model = init_two_layer_model(32*32*3, 100, 10) # input size, hidden size, number of classes
trainer = ClassifierTrainer()
best_model, loss_history, train_acc, val_acc = trainer.train(X_train, y_train, X_val, y_val,
                                             model, two_layer_net,
                                             num_epochs=5, reg=1.0,
                                             momentum=0.9, learning_rate_decay = 0.95,
                                             learning_rate=1e-5, verbose=True)


```

    starting iteration  0
    Finished epoch 0 / 5: cost 8.131793, train: 0.101000, val 0.117000, lr 1.000000e-05
    Finished epoch 1 / 5: cost 3.664248, train: 0.461000, val 0.399000, lr 9.500000e-06
    starting iteration  500
    Finished epoch 2 / 5: cost 3.184146, train: 0.455000, val 0.430000, lr 9.025000e-06
    starting iteration  1000
    Finished epoch 3 / 5: cost 3.131715, train: 0.494000, val 0.473000, lr 8.573750e-06
    starting iteration  1500
    Finished epoch 4 / 5: cost 2.673492, train: 0.547000, val 0.467000, lr 8.145063e-06
    starting iteration  2000
    Finished epoch 5 / 5: cost 2.565988, train: 0.546000, val 0.480000, lr 7.737809e-06
    finished optimization. best validation accuracy: 0.480000
    

# 训练过程监控
我们需要确保训练是正常进行的，你可以通过以下的办法去了解训练的状态：<br>
1）绘出随迭代进行的损失值变化，我们希望是逐步减小的<br>
2）可视化第一层的权重


```python
# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(loss_history)
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(train_acc)
plt.plot(val_acc)
plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
```




    <matplotlib.text.Text at 0x10ab78b90>




![png](output_21_1.png)



```python
from nn.vis_utils import visualize_grid

# 可视化权重

def show_net_weights(model):
    plt.imshow(visualize_grid(model['W1'].T.reshape(-1, 32, 32, 3), padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

show_net_weights(model)
```


![png](output_22_0.png)


# 调优参数

**上面的图告诉我们什么？**. 我们看到loss的下降近乎是线性的，这预示着可能__我们的学习率设得太小了__。如果训练和交叉验证集上的准确率差别又不是特别大，也可能说明模型的__容量（学习能力）__很有限，可以提高隐层的结点个数，不过话说回来，如果隐层节点个数取得太多，训练集和交叉验证集上可能准确率差别就会很大了，这有可能说明是过拟合了。

**调优**. 恩，你也听好多人吐槽过，说神经网络其实就是一个调参的活，这个，怎么说呢，有时候人家说的也没错。我们会对隐层结点个数，学习率，训练轮数和正则化参数进行优选。

**关于准确率**. 在现在的这个图片数据集上，我们应该至少要取得50%以上的准确率，不然肯定是哪块出问题了，得回过头去检查一下咯。


```python
from nn.classifiers.neural_net import init_two_layer_model
from nn.classifier_trainer import ClassifierTrainer

best_model = None # 存储交叉验证集上拿到的最好的结果
best_val_acc = -1
# 很不好意思，这里直接列了一堆参数，然后用for循环做的cross-validation
learning_rates = [1e-5, 5e-5, 1e-4]
model_capacitys = [200, 300, 500, 1000]
regularization_strengths = [1e0, 1e1]
results = {}
verbose = True

for hidden_size in model_capacitys:
    for lr in learning_rates:
        for reg in regularization_strengths:
            if verbose: 
                print "Trainging Start: "
                print "lr = %e, reg = %e, hidden_size = %e" % (lr, reg, hidden_size)

            model = init_two_layer_model(32*32*3, hidden_size, 10)
            trainer = ClassifierTrainer()
            output_model, loss_history, train_acc, val_acc = trainer.train(X_train, y_train, X_val, y_val,
                                             model, two_layer_net,
                                             num_epochs=5, reg=1.0,
                                             momentum=0.9, learning_rate_decay = 0.95,
                                             learning_rate=lr)


            results[hidden_size, lr, reg] = (loss_history, train_acc, val_acc)

            if verbose: 
                print "Training Complete: "
                print "Training accuracy = %f, Validation accuracy = %f " % (train_acc[-1], val_acc[-1])

            if val_acc[-1] > best_val_acc:
                best_val_acc = val_acc[-1]
                best_model = output_model
        
print 'best validation accuracy achieved during cross-validation: %f' % best_val_acc
```

    Trainging Start: 
    lr = 1.000000e-05, reg = 1.000000e+00, hidden_size = 2.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.606000, Validation accuracy = 0.508000 
    Trainging Start: 
    lr = 1.000000e-05, reg = 1.000000e+01, hidden_size = 2.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.590000, Validation accuracy = 0.494000 
    Trainging Start: 
    lr = 5.000000e-05, reg = 1.000000e+00, hidden_size = 2.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.570000, Validation accuracy = 0.497000 
    Trainging Start: 
    lr = 5.000000e-05, reg = 1.000000e+01, hidden_size = 2.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.571000, Validation accuracy = 0.512000 
    Trainging Start: 
    lr = 1.000000e-04, reg = 1.000000e+00, hidden_size = 2.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.569000, Validation accuracy = 0.497000 
    Trainging Start: 
    lr = 1.000000e-04, reg = 1.000000e+01, hidden_size = 2.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.524000, Validation accuracy = 0.500000 
    Trainging Start: 
    lr = 1.000000e-05, reg = 1.000000e+00, hidden_size = 3.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.610000, Validation accuracy = 0.516000 
    Trainging Start: 
    lr = 1.000000e-05, reg = 1.000000e+01, hidden_size = 3.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.603000, Validation accuracy = 0.508000 
    Trainging Start: 
    lr = 5.000000e-05, reg = 1.000000e+00, hidden_size = 3.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.565000, Validation accuracy = 0.522000 
    Trainging Start: 
    lr = 5.000000e-05, reg = 1.000000e+01, hidden_size = 3.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.581000, Validation accuracy = 0.503000 
    Trainging Start: 
    lr = 1.000000e-04, reg = 1.000000e+00, hidden_size = 3.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.537000, Validation accuracy = 0.480000 
    Trainging Start: 
    lr = 1.000000e-04, reg = 1.000000e+01, hidden_size = 3.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.506000, Validation accuracy = 0.509000 
    Trainging Start: 
    lr = 1.000000e-05, reg = 1.000000e+00, hidden_size = 5.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.645000, Validation accuracy = 0.510000 
    Trainging Start: 
    lr = 1.000000e-05, reg = 1.000000e+01, hidden_size = 5.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.594000, Validation accuracy = 0.532000 
    Trainging Start: 
    lr = 5.000000e-05, reg = 1.000000e+00, hidden_size = 5.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.628000, Validation accuracy = 0.513000 
    Trainging Start: 
    lr = 5.000000e-05, reg = 1.000000e+01, hidden_size = 5.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.583000, Validation accuracy = 0.521000 
    Trainging Start: 
    lr = 1.000000e-04, reg = 1.000000e+00, hidden_size = 5.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.536000, Validation accuracy = 0.509000 
    Trainging Start: 
    lr = 1.000000e-04, reg = 1.000000e+01, hidden_size = 5.000000e+02
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.570000, Validation accuracy = 0.500000 
    Trainging Start: 
    lr = 1.000000e-05, reg = 1.000000e+00, hidden_size = 1.000000e+03
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.653000, Validation accuracy = 0.513000 
    Trainging Start: 
    lr = 1.000000e-05, reg = 1.000000e+01, hidden_size = 1.000000e+03
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.630000, Validation accuracy = 0.528000 
    Trainging Start: 
    lr = 5.000000e-05, reg = 1.000000e+00, hidden_size = 1.000000e+03
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.600000, Validation accuracy = 0.527000 
    Trainging Start: 
    lr = 5.000000e-05, reg = 1.000000e+01, hidden_size = 1.000000e+03
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.580000, Validation accuracy = 0.511000 
    Trainging Start: 
    lr = 1.000000e-04, reg = 1.000000e+00, hidden_size = 1.000000e+03
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.549000, Validation accuracy = 0.522000 
    Trainging Start: 
    lr = 1.000000e-04, reg = 1.000000e+01, hidden_size = 1.000000e+03
    starting iteration  0
    starting iteration  500
    starting iteration  1000
    starting iteration  1500
    starting iteration  2000
    Training Complete: 
    Training accuracy = 0.533000, Validation accuracy = 0.508000 
    best validation accuracy achieved during cross-validation: 0.532000
    


```python
# 可视化参数权重
show_net_weights(best_model)

# 在测试集上看准确率
scores_test = two_layer_net(X_test, best_model)
print 'Test accuracy: ', np.mean(np.argmax(scores_test, axis=1) == y_test)
```


![png](output_25_0.png)


    Test accuracy:  0.525
    


```python
total_num = len(results)
for i, (hsize, lr, reg) in enumerate(sorted(results)):
    loss_history, train_acc, val_acc = results[hsize, lr, reg]
    
    if val_acc[-1] > 0.5: 
        plt.figure(i)
        plt.title('hidden size {0} lr {1} reg {2} train accuracy'.format(hsize, lr, reg))
        
        plt.subplot(2, 1, 1)
        plt.plot(loss_history)
        plt.title('Loss history')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        plt.subplot(2, 1, 2)
        plt.plot(train_acc)
        plt.plot(val_acc)
        plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
        plt.xlabel('Epoch')
        plt.ylabel('Clasification accuracy')

```


![png](output_26_0.png)



![png](output_26_1.png)



![png](output_26_2.png)



![png](output_26_3.png)



![png](output_26_4.png)



![png](output_26_5.png)



![png](output_26_6.png)



![png](output_26_7.png)



![png](output_26_8.png)



![png](output_26_9.png)



![png](output_26_10.png)



![png](output_26_11.png)



![png](output_26_12.png)



![png](output_26_13.png)



![png](output_26_14.png)



![png](output_26_15.png)



![png](output_26_16.png)



![png](output_26_17.png)


# 在测试集上看看效果
神经网络训练完了，咱们需要在测试集上看看效果


```python
scores_test = two_layer_net(X_test, best_model)
print 'Test accuracy: ', np.mean(np.argmax(scores_test, axis=1) == y_test)
```

    Test accuracy:  0.525
    


```python

```
