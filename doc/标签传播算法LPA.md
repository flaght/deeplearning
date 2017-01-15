# 标签传播算法LPA

标签（空格分隔）： SPARK机器学习

---

##1. 概念与原理
标签传播算法（label propagation）是一种半监督学习，即有少量的labeled的数据和大量的unlabeled的数据。(注意，该算法允许大部分的数据无label,可以只有少数几个有label)
半监督学习算法会充分的利用unlabeled数据来捕捉我们整个数据的潜在分布。它基于三大假设：

       1）Smoothness平滑假设：相似的数据具有相同的label。

       2）Cluster聚类假设：处于同一个聚类下的数据具有相同label。

       3）Manifold流形假设：处于同一流形结构下的数据具有相同label。

标签传播算法（label propagation）的核心思想非常简单：相似的数据应该具有相同的label。LP算法包括两大步骤：1）构造相似矩阵；2）勇敢的传播吧。


##2.参考文献
参考博客
http://blog.csdn.net/zouxy09/article/details/49105265
参考论文：
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf
wiki:
https://en.wikipedia.org/wiki/Label_Propagation_Algorithm

##3.scala程序逻辑
###3.1 创建顶点
LPA是基于图计算的，因此需要创建“顶点”与“边”，从而组成一个“图”来实现运算

我们先生成一组简单的测试数据，假设有五个实体对象（1,2,3,4,5），也就是5个顶点，其中假设我们已经知道1和4的label分别为0和1，而2,3和5的label我们无从知晓，是需要通过LPA算法去预测的。这些需要有待预测对象，在一开始可以随便赋予一个类别，无论你赋予它什么类别，在迭代之后，它最终会归于与它相似的类别中。

在不断迭代中，那些已知类别的点在每次迭代完之后需要保持原有的label,因此对是否已知label也需要打上标签，1为已知label的点,0为我们初始设定的待预测label的点

故创建顶点数据如下，第一个元素为id(每个顶点的唯一标识），第二个元素为对应的姓名（就是实际上的对象），第三个元素是是否为已知label的标记，第四个元素是label.
```
//创建顶点的测试集
val labelPoint = sc.parallelize(Array((1,Lucy,1,1),(2,Lily,0,1),(3,Hanmeimei,0,0),(4,Liming,1,0),(Julia,127,0,0)))
                   .map(x => (x._1.toLong, x._2.toString, x._3.toInt, x._4.toInt))

//创建顶点
val newsVertex: RDD[(VertexId, String)] = labelPoint.map(x => (x._1, x._2))
```

###3.2 创建边
创建边，即创建点与点之间的相似性。相似性越高，则边的权重越大。假设根据各个对象之间的特征我们计算出了他们的相似性（即权重，权重的计算方法很多）。如下，（1,2,0.8）表示id为1和2的两个点的权重为0.8
```
//创建边的测试集
val similarity = sc.parallelize(Array((1,2,0.8),(1,3,0.7),(1,4,0.2),(1,5,0.1),(2,3,0.9),(2,4,0.1),(2,5,0.05),(3,4,0.1),(3,5,0.1),(4,5,0.99)))

//创建边
val totalSimilarityTemp = similarity.union(similarity.map(x => (x._2, x._1, x._3)))
val totalSimilarity = totalSimilarityTemp.union(totalSimilarityTemp.map(x => (x._1, x._1, 1.0)).distinct)
val edgeRecord: RDD[Edge[Double]] = totalSimilarity.map(x => Edge(x._1, x._2, x._3.toDouble))
```

###3.3 创建图
将刚刚创建的顶点和边传入Graph()创建图
```
val graph = Graph(newsVertex, edgeRecord)
```

###3.4 创建标签矩阵
标签矩阵由已知标签点与未知标签点组成.
```
val labelMatrixTemp = labelPoint.map(x => ((x._1, x._3, x._4),(0.0, 0.0)))
val labelMatrix = labelMatrixTemp.map{case(k,v) =>
  if (k._3 == 1) {
    (k, (0.0, 1.0))
   } else {
    (k, (1.0, 0.0))
   }
  }
```

###3.5 创建传播概率
（1）首先，从我们之前建好的图中提取权重,如下weight表示（出发点，目的点，权重）
（2）再者，需要做行的归一化，使每个权重都除以它所在行的和。
```
val weight = graph.triplets.map(x => ((x.srcId.toLong, x.dstId.toLong), x.attr))

val rowSum = weight.map(x => (x._1._1, x._2)).reduceByKey(_+_)
val propagationPro = weight.map(x => (x._1._1, (x._1._2, x._2))).join(rowSum)
  .map(x => ((x._1,x._2._1._1),x._2._1._2 / x._2._2))
```

###3.6 迭代运算
新标签矩阵 = 传播概率*标签矩阵,方法multiplyMatrix实现了将传播概率与标签矩阵相乘的运算，相乘之后会产生新的标签矩阵，需要将已知label的点重新赋予它原来的label.然后进入下一轮迭代相乘。
```
/**
  * 传播概率与标签矩阵相乘
  * 
  * @param W 传播概率
  * @param Y 标签矩阵
  * @return 新的标签矩阵
  */
def multiplyMatrix(W: RDD[((Long, Long), Double)], Y: RDD[((Long, Int, Int), (Double, Double))]) = {

    val tempW = W.map(x => (x._1._2, (x._1._1, x._2)))
    val tempY = Y.map(x => (x._1._1, (x._1._2, x._2)))
    val matrixTemp = tempW.join(tempY).map(x => ((x._2._1._1, x._1), (x._2._1._2 * x._2._2._2._1,     x._2._1._2 * x._2._2._2._2, x._2._2._1)))

    val tempResult = matrixTemp.map(x => (x._1._1, (x._2._1, x._2._2))).reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
    
    val resultLabel = tempResult.join(Y.map(x => (x._1._1, (x._1._2, x._1._3)))).map(x => {
    
      val total = x._2._1._1 + x._2._1._2
      
      if (x._2._2._1 == 1) {
        if (x._2._2._2 == 1) {
          ((x._1, x._2._2._1, x._2._2._2), (0.0, 1.0))
      } else {
        ((x._1, x._2._2._1, x._2._2._2), (1.0, 0.0))
      }
    } else {
        ((x._1, x._2._2._1, x._2._2._2), (x._2._1._1 / total, x._2._1._2 / total))
      }

    })

    resultLabel
  }
```

```
var updateLabelMatrix = multiplyMatrix(weight, labelMatrix)
```
###3.7 循环迭代至收敛
```
 //循环相乘产生新的标签矩阵
 ITERATION = 50
 
 for (i <- 1 to ITERATION) {

   updateLabelMatrix = multiplyMatrix(weight, updateLabelMatrix)

   val labelRecordAfter = updateLabelMatrix.map(x => (x._1._1, x._2._2))

    }
    
 //赋予最大概率的标签
 val finalLabel = updateLabelMatrix.map{case(k, v) =>
   if (v._1 >= 0.5) {
     (k, 0.0)
   } else {
     (k, 1.0)
   }
 }
```
打印结果：
```
finalLabel.collect().foreach(println)

((4,1,0),0.0)
((1,1,1),1.0)
((5,0,0),0.0)
((2,0,1),1.0)
((3,0,0),1.0)
```



