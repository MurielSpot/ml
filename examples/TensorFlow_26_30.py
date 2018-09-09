'''
26:梯度下降可视化；调参。
27:全局最小值，局部最小值，调参。
28:
'''

# example 26 #########################################################
# 梯度下降可视化；调参。
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LR=0.1
REAL_PARAMS=[1.2,2.5]
# 列举多个参数，每次训练可以选择不同的参数，即调参。
INIT_PARAMS=[[5,4],
             [5,1],
             [2,4.5]][2]#即[2, 4.5]。
#print("initparams:",INIT_PARAMS)

# 200个数据的x轴坐标。
x=np.linspace(-1,1,200,dtype=np.float32)

# test 1
# 200个理想数据的y轴坐标。
y_fun=lambda a,b:a*x+b
# 定义公式，用来拟合a，b参数。
tf_y_fun=lambda a,b:a*x+b

# 噪声。
noise=np.random.randn(200)/10
# 生成真实数据的y轴坐标。
y=y_fun(*REAL_PARAMS)+noise
# 把真实数据画出来。
plt.scatter(x,y)

# 给a，b赋初始值，2,4.5。
a,b=[tf.Variable(initial_value=p,dtype=tf.float32) for p in INIT_PARAMS]
#print("a,b:",a,b)

pred=tf_y_fun(a,b)
mse=tf.reduce_mean(tf.square(y-pred))
train_op=tf.train.GradientDescentOptimizer(LR).minimize(mse)

# 存训练时，参数的变化，用来画图。
a_list,b_list,cost_list=[],[],[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(100):
        a_,b_,mse_=sess.run([a,b,mse])
        a_list.append(a_);b_list.append(b_);cost_list.append(mse_)
        result,_=sess.run([pred,train_op])

    # visualization codes:
    print('a=', a_, 'b=', b_, "mse=",mse_)
    plt.figure(1)
    plt.scatter(x, y, c='b')    # plot data
    plt.plot(x, result, 'r-', lw=2)   # plot line fitting
    # 3D cost figure
    fig = plt.figure(2); ax = Axes3D(fig)
    a3D, b3D = np.meshgrid(np.linspace(-2, 7, 30), np.linspace(-2, 7, 30))  # parameter space
    cost3D = np.array([np.mean(np.square(y_fun(a_, b_) - y)) for a_, b_ in zip(a3D.flatten(), b3D.flatten())]).reshape(a3D.shape)
    ax.plot_surface(a3D, b3D, cost3D, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), alpha=0.5)
    ax.scatter(a_list[0], b_list[0], zs=cost_list[0], s=300, c='r')  # initial parameter place
    ax.set_xlabel('a'); ax.set_ylabel('b')
    ax.plot(a_list, b_list, zs=cost_list, zdir='z', c='r', lw=3)    # plot 3D gradient descent
    plt.show()

    
# example 27 #########################################################
# 全局最小值，局部最小值，调参。
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LR=0.1
REAL_PARAMS=[1.2,2.5]
INIT_PARAMS=[[5,4],
             [5,1],
             [2,4.5]][0]#使用第一组参数结果很差，使用第三组结果不错。

x=np.linspace(-1,1,200,dtype=np.float32)

# 画图时注意到有：local minimum和global minimum。
y_fun=lambda a,b:np.sin(b*np.cos(a*x))
# 注意训练时用的是tf的sin和cos。
tf_y_fun=lambda a,b:tf.sin(b*tf.cos(a*x))

noise=np.random.randn(200)/10
y=y_fun(*REAL_PARAMS)+noise
plt.scatter(x,y)

a,b=[tf.Variable(initial_value=p,dtype=tf.float32) for p in INIT_PARAMS]

pred=tf_y_fun(a,b)
mse=tf.reduce_mean(tf.square(y-pred))
train_op=tf.train.GradientDescentOptimizer(LR).minimize(mse)

# 存训练时，参数的变化，用来画图。
a_list,b_list,cost_list=[],[],[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(100):
        a_,b_,mse_=sess.run([a,b,mse])
        a_list.append(a_);b_list.append(b_);cost_list.append(mse_)
        result,_=sess.run([pred,train_op])

    # visualization codes:
    print('a=', a_, 'b=', b_, "mse=",mse_)
    plt.figure(1)
    plt.scatter(x, y, c='b')    # plot data
    plt.plot(x, result, 'r-', lw=2)   # plot line fitting
    # 3D cost figure
    fig = plt.figure(2); ax = Axes3D(fig)
    a3D, b3D = np.meshgrid(np.linspace(-2, 7, 30), np.linspace(-2, 7, 30))  # parameter space
    cost3D = np.array([np.mean(np.square(y_fun(a_, b_) - y)) for a_, b_ in zip(a3D.flatten(), b3D.flatten())]).reshape(a3D.shape)
    ax.plot_surface(a3D, b3D, cost3D, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), alpha=0.5)
    ax.scatter(a_list[0], b_list[0], zs=cost_list[0], s=300, c='r')  # initial parameter place
    ax.set_xlabel('a'); ax.set_ylabel('b')
    ax.plot(a_list, b_list, zs=cost_list, zdir='z', c='r', lw=3)    # plot 3D gradient descent
    plt.show()

# example 28 #########################################################


# example 29 #########################################################


# example 30 #########################################################
