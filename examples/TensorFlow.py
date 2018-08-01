# 每一个example可以单独运行

# example 1 #########################################################
#交互式使用
import tensorflow as tf

# 进入一个交互式的会话
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 使用初始化来初始化 Variable
x.initializer.run()

sub=tf.subtract(x,a)

# 输出结果
print(sub.eval())

# example 2 #########################################################
import tensorflow as tf
#import numpy  as np

# 构建图
# 创建一个常量 op, 产生一个1x2 矩阵,这个op被称作是一个节点,加到默认图中,构造器的返回值代表常量 op的返回值.
a = tf.constant([[3., 3.]])
b = tf.constant([[2.],[2.]])
product = tf.matmul(a,b)

# 启动一个图
# 现在已经创建好了一个 两个矩阵相乘并返回product结果的图。为了得到product，我们就必须在一个会话（Session）中启动这个一个图了。
# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数. 
# 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回矩阵乘法 op 的输出.
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
# 返回值 'result' 是一个 numpy `ndarray` 对象.
sess = tf.Session()
result = sess.run(product)
print(result)

# 任务完成后就需要关闭
sess.close()

# 有时候会忘记写sess.close().这里我们可以使用系统的带的with来实现session的自动关闭。
# with tf.Session() as sess:
#      result = sess.run([product])
#      print result

# example 3 #########################################################
import tensorflow as tf
 
# 变量 Variable
# 创建一个变量，  初始化为标量 0
x = tf.Variable(0, name="counter")

# 创建一个operation, 其作用是使x 增加 1
one = tf.constant(1)
new_value = tf.add(x,one)

# tf.assign(A, new_number): 把A的值变为new_number
update = tf.assign(x, new_value)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
# init_op = tf.initialize_all_variables()已经不推荐使用了（2018.8）
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 运行 init_op
    sess.run(init_op)
    # 打印初始状态
    print(sess.run(x))
    for _ in range(3):
        sess.run(update)
        print(sess.run(x))

# example 4 #########################################################
import tensorflow as tf
 
# Fetch
# 为了取回操作的输出内容, 可以在使用 Session 对象的 run() 调用 执行图时, 传入一些 tensor, 这些 tensor 会帮助你取回结果. 
# 如取回多个 tensor:
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)
with tf.Session() as sess:
  result = sess.run([mul, intermed])
  print(result)

# example 5 #########################################################
import tensorflow as tf
 
# Feed
# feed 机制可以临时替代图中的任意操作中的 tensor 可以对图中任何操作提交补丁, 直接插入一个 tensor.
# feed 使用一个 tensor 值临时替换一个操作的输出结果. 
# 你可以提供 feed 数据作为 run() 调用的参数. feed 只在调用它的方法内有效, 方法结束, feed 就会消失. 
# 最常见的用例是将某些特殊的操作指定为 "feed" 操作, 标记的方法是使用 tf.placeholder() 为这些操作创建占位符.
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))

# example 6 #########################################################
#概率学中的逆概率
#什么是逆概率
#我们肯定知道正概率，举个例子就是，箱子里有5个黑球5个白球，那你随机拿到黑球和白球的概率都是50%，那现在我不知道箱子里有多少个黑球白球，那我通过不断的拿球应该如何确定箱子里有多少个黑球白球呢，这就是出名的逆概率
#其实机器学习很多时候也就是逆概率的问题，我有大量现实例子的情况下，让机器从这些例子中找到共同的特征，例如给一万张猫的图片给机器学习，然后找到共同的特征（两只耳朵，四只脚，有胡须，有毛，有尾巴等特征）
#根据逆概率的概念我们再举个其他场景

#y=Ax+B（A、B是常量），这是一条非常简单的数学方程式，有小学基础的人应该都知道。
#我现在有很多的x和y值，所以问题就是如何通过这些x和y值来得到A和B的值？
#接下来解决这个问题

import numpy as np
import tensorflow as tf

##构造数据
x_data=np.random.rand(100).astype(np.float32) #随机生成100个类型为float32的值
y_data=x_data*0.1+0.3  #定义方程式y=x_data*A+B
print(y_data)

##建立TensorFlow神经计算结构
weight=tf.Variable(tf.random_uniform([1],-1.0,1.0)) 
biases=tf.Variable(tf.zeros([1]))     
y=weight*x_data+biases
print(y)

#判断与正确值的差距
#tensorflow中有一类在tensor的某一维度上求值的函数。如：
#求最大值tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)
#求平均值tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
#参数1--input_tensor:待求值的tensor。
#参数2--reduction_indices:在哪一维上求解。
loss=tf.reduce_mean(tf.square(y-y_data))

#根据差距进行反向传播修正参数
#class tf.train.GradientDescentOptimizer
#这个类是实现梯度下降算法的优化器。(结合理论可以看到，这个构造函数需要的一个学习率就行了)
#__init__(learning_rate, use_locking=False,name=’GradientDescent’)
#作用：创建一个梯度下降优化器对象 
#参数： 
#learning_rate: A Tensor or a floating point value. 要使用的学习率 
#use_locking: 要是True的话，就对于更新操作（update operations.）使用锁 
#name: 名字，可选，默认是”GradientDescent”.
optimizer=tf.train.GradientDescentOptimizer(0.5)

#建立训练器
train=optimizer.minimize(loss)

#初始化TensorFlow训练结构
init=tf.global_variables_initializer() 

#建立TensorFlow训练会话
sess=tf.Session()

#将训练结构装载到会话中
sess.run(init)

for step in range(400): #循环训练400次
     sess.run(train)  #使用训练器根据训练结构进行训练
     if  step%20==0:  #每20次打印一次训练结果
        print(step,sess.run(weight),sess.run(biases)) #训练次数，A值，B值

# example 7 #########################################################


# example 8 #########################################################


# example 9 #########################################################
