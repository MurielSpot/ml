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
# 可视化,使用tensorboard.

import tensorflow as tf
import numpy as np

def add_layer(inputs,in_size,out_size,activation_func=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            # in_size行,out_size列。
            Weights=tf.Variable(tf.random_normal([in_size,out_size]))
        with tf.name_scope('biases'):
            # 设置biases全为0.1。
            biases=tf.Variable(tf.zeros([1,out_size])+0.1)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(inputs,Weights)+biases
        
        #激活函数不明确给它命名,它也会默认有相应的名字,所以可以不用tf.name_scope.
        if activation_func is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_func(Wx_plus_b)
        return outputs

with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')

#第一层隐藏层
l1=add_layer(xs,1,10,activation_func=tf.nn.relu)

#输出层
prediction=add_layer(l1,10,1,activation_func=None)

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
with tf.name_scope('train_step'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    #将整个框架,写入文件,然后再用浏览器读这个文件,才能看到整个框架结构.
    #tensorflow 新版取消了tf.train.SummaryWriter()，换成使用tf.summary.FileWriter()
    writer=tf.summary.FileWriter("logs/",sess.graph)
    sess.run(init)

'''
文件存到文件夹里之后.
在命令行里输入:
tensorboard --logdir=文件所在文件夹路径
打开tensorboard之后,命令行里会提示可以通过哪个网址来查看整个神经网络框架的图像.
'''

# example 10 #########################################################
# 可视化,使用tensorboard.

import tensorflow as tf
import numpy as np

def add_layer(inputs,in_size,out_size,n_layer,activation_func=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope(layer_name+'_scope'):
        with tf.name_scope('weights_scope'):
            # in_size行,out_size列。
            Weights=tf.Variable(tf.random_normal([in_size,out_size]))
            #tf.histogram_summary()改为：tf.summary.histogram()
            # 了解weights变量的变化情况.
            tf.summary.histogram('/Weights',Weights)
        with tf.name_scope('biases_scope'):
            # 设置biases全为0.1。
            biases=tf.Variable(tf.zeros([1,out_size])+0.1)
            tf.summary.histogram('/biases',biases)
        with tf.name_scope('Wx_plus_b_scope'):
            Wx_plus_b=tf.matmul(inputs,Weights)+biases
        
        #激活函数不明确给它命名,它也会默认有相应的名字,所以可以不用tf.name_scope.
        if activation_func is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_func(Wx_plus_b)
        tf.summary.histogram('/outputs',outputs)
        return outputs

# make up some real data.
x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

with tf.name_scope('inputs_scope'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')

#第一层隐藏层,注意前面定义函数时增加了n_layer参数.
l1=add_layer(xs,1,10,n_layer=1,activation_func=tf.nn.relu)

#输出层
prediction=add_layer(l1,10,1,n_layer=2,activation_func=None)

with tf.name_scope('loss_scope'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    #与tf.summary.histogram()不同点在于,它不会在tensorboard的histogram里显示,而是在events里显示.
    tf.summary.scalar('loss',loss)
with tf.name_scope('train_step_scope'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    #将前面的summary都合并起来.
    merged=tf.summary.merge_all()
    writer=tf.summary.FileWriter("logs/",sess.graph)
    sess.run(init)

    #训练100次.
    for i in range(100):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i%5==0:
            result=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
            writer.add_summary(result,i)

# example 11 #########################################################
# 分类classification.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
# 如果电脑上没有这个数据包,这句会帮你从网上下载下来.
# './MNIST_data'可以改成数据在电脑上的完整路径.
mnist=input_data.read_data_sets('./MNIST_data',one_hot=True)

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b,)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction#设置prediction为全局变量.

    #将v_xs传到prediction里面,生成预测值.
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    #预测的值与实际值是否相同.
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    #计算准确率.
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

# define placeholder for inputs to network
# [None,784]不规定有多少个sample,但规定每个sample的大小是784.
xs=tf.placeholder(tf.float32,[None,784])
# 每个sample(此代码里表示输出)大小为10.
ys=tf.placeholder(tf.float32,[None,10])

# add output layer
prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)

# the error between prediction and real data.
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        #设置训练时每一批次大小为100.
        batch_xs,batch_ys=mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
        if i%5==0:
            print(compute_accuracy(mnist.test.images,mnist.test.labels))

# example 12 #########################################################
# 训练集，测试集
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data.
digits=load_digits()
X=digits.data#从0到9的图片数据。
y=digits.target
y=LabelBinarizer().fit_transform(y)#把y变成二进制形式。
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3)

def add_layer(inputs,in_size,out_size,layer_name,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b,)
    #不加下面这句好像会报错。
    tf.summary.histogram(layer_name+'/outputs',outputs)
    return outputs

# define placeholder for inputs to network
xs=tf.placeholder(tf.float32,[None,64])#8*8
ys=tf.placeholder(tf.float32,[None,10])

# add output layer
l1=add_layer(xs,64,100,'l1',activation_function=tf.nn.tanh)
prediction=add_layer(l1,100,10,'l2',activation_function=tf.nn.softmax)

# the error between prediction and real data.
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
tf.summary.scalar('loss',cross_entropy)
train_step=tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged=tf.summary.merge_all()

    # 分别记录train和test的summary。
    train_writer=tf.summary.FileWriter('logs/train',sess.graph)
    test_writer=tf.summary.FileWriter('logs/test',sess.graph)

    for i in range(100):
        sess.run(train_step,feed_dict={xs:X_train,ys:y_train})
        if i%5==0:
            #record loss
            train_result=sess.run(merged,feed_dict={xs:X_train,ys:y_train})
            test_result=sess.run(merged,feed_dict={xs:X_train,ys:y_train})
            train_writer.add_summary(train_result,i)
            test_writer.add_summary(test_result,i)

# 可以用tensorboard直接看logs里的两个文件夹test，train里的图像。

# example 13 #########################################################
# dropout

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data.
digits=load_digits()
X=digits.data#从0到9的图片数据。
y=digits.target
y=LabelBinarizer().fit_transform(y)#把y变成二进制形式。
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3)

def add_layer(inputs,in_size,out_size,layer_name,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    # dropout加在这里。
    Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob)
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b,)
    #不加下面这句好像会报错。
    tf.summary.histogram(layer_name+'/outputs',outputs)
    return outputs

# define placeholder for inputs to network
keep_prob=tf.placeholder(tf.float32)#结果不被dropout的概率。
xs=tf.placeholder(tf.float32,[None,64])#8*8
ys=tf.placeholder(tf.float32,[None,10])

# add output layer
l1=add_layer(xs,64,50,'l1',activation_function=tf.nn.tanh)
prediction=add_layer(l1,50,10,'l2',activation_function=tf.nn.softmax)

# the error between prediction and real data.
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
tf.summary.scalar('loss',cross_entropy)
train_step=tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged=tf.summary.merge_all()

    # 分别记录train和test的summary。
    train_writer=tf.summary.FileWriter('logs/train',sess.graph)
    test_writer=tf.summary.FileWriter('logs/test',sess.graph)

    for i in range(100):
        sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.5})
        if i%5==0:
            #record loss
            train_result=sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
            test_result=sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1 })
            train_writer.add_summary(train_result,i)
            test_writer.add_summary(test_result,i)

# example 14 #########################################################
# cnn.没跑出来，一方面因为资源不足，另一方面，好像最后一行print精确度的函数也报错了。

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#跑的时候下面的文件地址要改一下。
mnist=input_data.read_data_sets(r'D:……………………\MNIST_data',one_hot=True)

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

def weight_variable(shape):
    #产生随机变量。
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    # stride [1,x_movement,y_movement,1],网上有人说分别是stride [batch,height,width,channels]
    # must have strides[0]=strides[4]=1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    # stride [1,x_movement,y_movement,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# define placeholder for inputs to network
keep_prob=tf.placeholder(tf.float32)#结果不被dropout的概率。
xs=tf.placeholder(tf.float32,[None,784])#28*28
ys=tf.placeholder(tf.float32,[None,10])
#-1指不管sample一共有多少个，28*28的图片，channel为1.
x_image=tf.reshape(xs,[-1,28,28,1])

#conv1 layer
W_conv1=weight_variable([5,5,1,32])#patch 5*5,in size 1，out size 32.
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#output size 28*28*32.
h_pool1=max_pool_2x2(h_conv1)#output size 14*14*32.

#conv2 layer
W_conv2=weight_variable([5,5,32,64])#patch 5*5,in size 32，out size 64.
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#output size 14*14*64.
h_pool2=max_pool_2x2(h_conv2)#output size 7*7*64.

#func1 layer
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

h_poo2_flat=tf.reshape(h_pool2,[-1,7*7*64])#[n_samples,7,7,64]->[n_samples,7*7*64].
h_fc1=tf.nn.relu(tf.matmul(h_poo2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#func2 layer
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# the error between prediction and real data.
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        batch_xs,batch_ys=mnist.train.next_batch(10)
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
        print(compute_accuracy(mnist.test.images,mnist.test.labels))

# example 15 #########################################################
# 将模型保存到文件。
import tensorflow as tf

# save to file
W=tf.Variable([[1,2,3],[4,5,6]],dtype=tf.float32,name='weights')
b=tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')

init=tf.global_variables_initializer()

saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path=saver.save(sess,"./save_net.ckpt")

# example 16 #########################################################
# 接上面的例子，从文件中读取模型。
# 注意，目前TensorFlow只能保存参数，不能保存神经网络框架，如果从文件中读取参数的时候需要重新定义框架。
import tensorflow as tf
import numpy as np

# remember to define the same dtype and shape when restore.
W=tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name="weights")
b=tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name="biases")

# not need init step
saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"./save_net.ckpt")
    print("weights:",sess.run(W))
    print("biases:",sess.run(b))

# example 17 #########################################################
# 自加1程序：
import tensorflow as tf

x=tf.Variable(0,tf.int32)
increment=tf.assign(x,x+1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        print(sess.run(increment))#输出1到10.

# example 18 #########################################################
# fetch，可以用列表让sess.run一次运行多个op。
import tensorflow as tf

x=tf.Variable(0,tf.int32)
y=tf.constant([1,1],tf.int32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    print(sess.run([x,y]))#用列表可以一次运行多个op。
    # [0, array([1, 1])]

# example 19 #########################################################


# example 20 #########################################################


# example 21 #########################################################


# example 22 #########################################################


# example 23 #########################################################


# example 24 #########################################################


# example 25 #########################################################


# example 26 #########################################################
