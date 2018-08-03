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
# 打印变量名和值
print(x.name)

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
# placeholder需要指明数据类型，一般tf常用float32.
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
  print(sess.run(output, feed_dict={input1:[7.7], input2:[2.]}))
  # 上面两句分别输出：[array([14.], dtype=float32)]和[15.4]。

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

sess.close()

# example 7 #########################################################
import tensorflow as tf

matrix1=tf.constant([[3,3]])
matrix2=tf.constant([[2],
                     [2]])

#matrix multiply,numpy中矩阵乘法为 np.dot(matrix1,matrix2)
product=tf.matmul(matrix1,matrix2)

sess=tf.Session()
for i in range(5):
    matrix1=sess.run(product)
    print(matrix1)

sess.close()

'''
注意输出的结果：
[[12]]
[[12]]
[[12]]
[[12]]
[[12]]

for循环里的matrix1在第一次被改变之后，输出12，再次运行sess，结果还是12.
'''

# example 8 #########################################################
# 拟合模型与实际数据，并将实际数据和模型的曲线画出来。

import tensorflow as tf
import numpy as np

# 可视化
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_func=None):
    # in_size行,out_size列。
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    # 设置biases全为0.1。
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)

    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_func is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_func(Wx_plus_b)
    return outputs

'''
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
在指定的间隔内返回均匀间隔的数字。
返回num均匀分布的样本，在[start, stop]。
'''
x_data=np.linspace(-1,1,300)[:,np.newaxis]
print("未使用[:,np.newaxis]时，np.linspace(-1,1,300):",np.linspace(-1,1,300))
print("使用[:,np.newaxis]后：",x_data)

'''
numpy.random.normal(loc=0.0, scale=1.0, size=None)

参数的意义为：
loc：float
    此概率分布的均值（对应着整个分布的中心centre）
scale：float
    此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
size：int or tuple of ints
    输出的shape，默认为None，只输出一个值
'''
# 使用noise让y对应的点不要完全符合x的平方的规律，即让输出的点更随机。
noise=np.random.normal(0,0.05,x_data.shape)
print("np.random.normal(0,0.05,x_data.shape)输出为：",noise)

y_data=np.square(x_data)-0.5+noise

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

#第一层隐藏层
l1=add_layer(xs,1,10,activation_func=tf.nn.relu)

#输出层
prediction=add_layer(l1,10,1,activation_func=None)

'''
#reduce_sum应该理解为压缩求和，用于降维
# 'x' is [[1, 1, 1]
#         [1, 1, 1]]
#求和
tf.reduce_sum(x) ==> 6
#按列求和
tf.reduce_sum(x, 0) ==> [2, 2, 2]
#按行求和
tf.reduce_sum(x, 1) ==> [3, 3]
#按照行的维度求和
tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
#行列求和
tf.reduce_sum(x, [0, 1]) ==> 6

A = np.array([[1,2], [3,4]])
with tf.Session() as sess:
	print sess.run(tf.reduce_mean(A))
	print sess.run(tf.reduce_mean(A, axis=0))
	print sess.run(tf.reduce_mean(A, axis=1))
输出：
2 #整体的平均值
[2 3] #按列求得平均
[1 3] #按照行求得平均
'''
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init=tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  # 画画
  fig=plt.figure()
  ax=fig.add_subplot(1,1,1)
  ax.scatter(x_data,y_data)
  plt.ion()#让图像绘制不停止，画完一个继续画下一个。
  plt.show()

  for step in range(1000):
    # 如果把一部分x_data传给xs，则可以实现小批量学习。
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if step%50==0:
        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))

        try:
            #要把之前画的线抹除之后，再画下一条。
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value=sess.run(prediction,feed_dict={xs:x_data,ys:y_data})
        # r-表示红色的线，lw表示宽度为5.
        lines=ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.1)


# example 9 #########################################################


# example 10 #########################################################


# example 11 #########################################################


# example 12 #########################################################

# example 13 #########################################################

# example 14 #########################################################

# example 15 #########################################################

# example 16 #########################################################

