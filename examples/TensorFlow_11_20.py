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