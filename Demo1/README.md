# 前提条件和准备工作

[参照官方对于机器学习入门的建议](https://developers.google.cn/machine-learning/crash-course/prereqs-and-prework)

主要是要求基本一定的 数学基础（本科课程能力）与python编程基础。你可以先刷一遍官方文档，也可以在后续的文章中，根据提到的知识点进行学习（对自己的知识体系进行查缺补漏）。



# TensorFlow 基本概念

[官方中文文档](http://wiki.jikexueyuan.com/project/tensorflow-zh/get_started/basic_usage.html)， 以下内容为个人笔记（概要）

使用 TensorFlow, 你必须明白 TensorFlow:

- 使用图 (graph) 来表示计算任务.
- 在被称之为 `会话 (Session)` 的上下文 (context) 中执行图.
- TensorFlow 程序使用 `tensor(张量)`数据结构来代表所有的数据。tensor 看作是一个 n 维的数组或列表. 一个 tensor 包含一个静态类型 rank, 和 一个 shape.
- 通过 `变量 (Variable)` 维护状态.
- 使用 feed 和 fetch 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据.



TensorFlow 是一个编程系统, 使用图来表示计算任务. 图中的节点被称之为 *op* (operation 的缩写). 一个 op 获得 0 个或多个 `Tensor`, 执行计算, 产生 0 个或多个 `Tensor`. 每个 Tensor 是一个类型化的多维数组. 例如, 你可以将一小组图像集表示为一个四维浮点数数组, 这四个维度分别是 `[batch, height, width, channels]`.

![mark](http://oz12t0u5u.bkt.clouddn.com/blog/180411/Ad1B689i2E.png?imageslim)



# 先看Demo

## 构建并启动图

```
import tensorflow as tf

# 创建常量 op
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])
# 创建一个矩阵乘法op， 把m1和m2传入
product = tf.matmul(m1,m2)
print(product)

# 定义一个会话，启动默认的图 (常用写法,无需手动关闭会话)
with tf.Session() as sess:
    # 执行以上的op
    result = sess.run(product)
    print(result)
```

- 构建图 ：的第一步, 是创建源 op (source op). 源 op 不需要任何输入, 例如 `常量 (Constant)`. 源 op 的输出被传递给其它 op 做运算.



- 启动图 ： 构造阶段完成后, 才能启动图. 启动图的第一步是创建一个 `Session` 对象, 如果无任何创建参数, 会话构造器将启动默认图.

## 常量与变量

```
# 创建常量 op
m1 = tf.constant([[3,3]])

# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter") 

# 启动图, 运行 op
with tf.Session() as sess:
  # 启动图后, 变量必须先经过`初始化` (init) op 初始化,
  sess.run(tf.initialize_all_variables())
  
  # to do something...
```



> 常量使用前(与我们平时的声明、定义变量有点不大一样)，必须进行初始化.



## 加减乘除

这里要求我们掌握 tf 的运算方法的用时，要求我们对矩阵运算有基本的了解，[参考资料](http://www2.edu-edu.com.cn/lesson_crs78/self/j_0022/soft/ch0605.html)。另外，代数相关知识建议看[deeplearning 第二章](https://github.com/exacity/deeplearningbook-chinese)

- 标量 （Scalar） ：一个单独的数
- 向量 （Vector） ：一列数
- 矩阵 （Matrix） ：一个二维数组
- 张量 （tensor） ：在某些情况下，我们会讨论坐标超过两维的数组。



> 注意： 矩阵乘法函数为：matmul

```
# 标量、向量
tf.add(2,3)        # 5
tf.subtract(3,2)   # 1
tf.multiply(2,3)   # 6

tf.add([3,3],[2,3])        # [5 6]
tf.subtract([3,3],[2,3])   # [1 0]
tf.multiply([3,3],[2,3])   # [6 9]

# 矩阵(2*2)、张量
tf.add([[3,3],[4,4]],[[2,1],[2,2]])        # [[5 4]  [6 6]]
tf.subtract([[3,3],[4,4]],[[2,1],[2,2]])   # [[1 2]  [2 2]]
tf.matmul([[3,3],[4,4]],[[2,1],[2,2]])     # [[12  9]  [16 12]]
```

###  

### 类型转换 

函数

```
# tensor `a` is [1.8, 2.2], dtype=tf.float
tf.cast(a, tf.int32) ==> [1, 2]  # dtype=tf.int32
```

为了让特定运算能运行，有时会对类型进行转换。例如

```
tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))   # 1
```









