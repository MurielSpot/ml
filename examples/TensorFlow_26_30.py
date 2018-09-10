'''
26:梯度下降可视化；调参。
27:全局最小值，局部最小值，调参。
28:cnn,图像分类。
29:
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
# cnn,图像分类。
# npy中读出784像素的图像，添加one-hot格式的标签，数据标签合并后打乱顺序，再将数据标签分开分别输出。
# 需要将数据变成[-1,28,28,1]格式传给卷积神经网络。-1表示自动计算出了图片有多少张，28,28是图片的行列，1对应一个filter（应该是这样，我还没仔细查）。
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

datasets_path=r"./ml_datasets/sketches/"

#每一类中取得多少个训练数据个数。
TRAIN_IMG_NUM=5000
#测试数据是紧接着训练数据的那些图片。
TEST_IMG_NUM=200

IMG_SIZE=28*28

LR=0.01
EPOCH=601
BATCH_SIZE=100#先不分批训练。
CLASS_NUM=3

'''

'''
def read_data(folder_path,train=True):
    data=[]
    labels=[]

    # 序号到类名的映射。
    stuff_index_to_class={}
    # 序号从0开始。
    stuff_now_index=0
    
    for file in os.listdir(folder_path):
        # 获得当前类别名。
        now_class=file.split(".npy")[0].split("_")[-1]
        # 序号 map to 类别名。
        stuff_index_to_class[stuff_now_index]=now_class

        if train:
            # imgs shape：[train+test img num,784]
            imgs=np.load(os.path.join(folder_path,file),"r")[:TRAIN_IMG_NUM]
            # labels shape: [train+test img num,class num]
            indexes=np.zeros((TRAIN_IMG_NUM,CLASS_NUM))
            indexes[:,stuff_now_index]=1
        else:
            imgs=np.load(os.path.join(folder_path,file),"r")[TRAIN_IMG_NUM:TRAIN_IMG_NUM+TEST_IMG_NUM]
            # labels shape: [train+test img num,class num]
            indexes=np.zeros((TEST_IMG_NUM,CLASS_NUM))
            indexes[:,stuff_now_index]=1

        # 拼接。
        if data!=[]:
            data=np.vstack([data,imgs])#要把结果赋值给data！
            labels=np.vstack([labels,indexes])
        else:
            data=imgs
            labels=indexes

        # 序号加一，序号数也就是当前一共有多少类别的数目。
        stuff_now_index+=1

    #print(np.shape(data))
    ## 可视化。
    #plt.show()
    #plt.ion()
    #cnt=len(data)
    #for i in range(cnt):
    #    plt.imshow(np.reshape(data[i],(28,28)),shape=(28,28))
    #    print(labels[i])
    #    plt.pause(0.5)

    total=np.hstack([data,labels])
    np.random.shuffle(total)
    data=total[:,:-CLASS_NUM]
    labels=total[:,-CLASS_NUM:]
    print(np.shape(data),np.shape(labels))

    return data,labels,stuff_index_to_class

def cnn_net(inputs):
    def cnn_layer(inputs,filters_num):
        initializer=tf.random_normal_initializer(0.0,0.2)
        return tf.layers.conv2d(inputs,filters=filters_num,kernel_size=(2,2),strides=(2,2),padding="same",kernel_initializer=initializer,activation=tf.nn.relu)

    def dense_layer(inputs,units_num):
        return tf.layers.dense(inputs,units=units_num,activation=tf.nn.sigmoid,use_bias=tf.constant_initializer(0.1),kernel_initializer=tf.random_normal_initializer(0.0,0.2))
    
    # inputs形状需要改成(28*28,1).
    inputs=tf.reshape(inputs,[-1,28,28,1])

    # cnn.
    # (28*28,1)=>(14*14,4)
    h1=cnn_layer(inputs,4)
    # (14*14,4)=>(7*7,8)
    h2=cnn_layer(h1,8)
    # (7*7,8)=>(4*4,16)
    h3=cnn_layer(h2,16)
    
    # flatten.
    h4=tf.layers.flatten(h3)

    # dense.
    h5=dense_layer(h4,20)
    h6=dense_layer(h5,20)
    h7=dense_layer(h6,3)

    # softmax.
    outputs=tf.nn.softmax(h7)
    
    return outputs

def main():
    # 测试数据和训练数据应该分别获得。
    train_data,train_labels,i2c=read_data(datasets_path)
    test_data,test_labels,_=read_data(datasets_path,train=False)

    x=tf.placeholder(dtype=tf.float32,shape=[None,784])
    y=tf.placeholder(dtype=tf.float32,shape=[None,3])

    outputs=cnn_net(x)

    loss=tf.losses.mean_squared_error(y,outputs)
    train_op=tf.train.AdamOptimizer(LR).minimize(loss)

    judge_class=tf.argmax(outputs,1)
    label_class=tf.argmax(y,1)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(judge_class,label_class),tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(EPOCH):
            _,_loss,_outputs=sess.run([train_op,loss,outputs],feed_dict={x:train_data,y:train_labels})
            
            if i%200==0:
                _accuracy=sess.run(accuracy,feed_dict={x:test_data,y:test_labels})
                print("epoch [%d]: loss:"%i,_loss,"test accuracy:",_accuracy)
                print("    outputs:\n",_outputs)
                print("\n")
                
main()

'''
输出：
#训练数据、标签格式。15000个图片。784个像素。三种类型（book，tree，bike）。
(15000, 784) (15000, 3)
#测试数据、标签格式。600个图片。784个像素。三种类型。
(600, 784) (600, 3)
#损失，准确率，神经网络的输出。神经网络最后一层是softmax，所以outputs一行中最大的那个位置对应图片最可能属于的那个类。
epoch [0]: loss: 0.2226727 test accuracy: 0.40666667
    outputs:
 [[0.30827367 0.33800027 0.353726  ]
 [0.30418524 0.34146315 0.3543516 ]
 [0.30357164 0.340107   0.35632136]
 ...
 [0.30496582 0.33879626 0.35623792]
 [0.30941522 0.33841622 0.35216853]
 [0.30271247 0.3420886  0.35519886]]


epoch [200]: loss: 0.10351374 test accuracy: 0.90833336
    outputs:
 [[0.5743267  0.21308106 0.21259218]
 [0.2121995  0.21610664 0.57169384]
 [0.21293351 0.57434046 0.21272604]
 ...
 [0.5743267  0.21308106 0.21259218]
 [0.21271977 0.21263716 0.5746431 ]
 [0.21591756 0.21190347 0.57217896]]


epoch [400]: loss: 0.100417405 test accuracy: 0.9116667
    outputs:
 [[0.575571   0.2122265  0.21220244]
 [0.21198188 0.21285914 0.57515895]
 [0.21206918 0.57470447 0.21322627]
 ...
 [0.5754642  0.21242398 0.21211186]
 [0.21203934 0.21227098 0.57568973]
 [0.21363223 0.21195468 0.57441306]]


epoch [600]: loss: 0.10061984 test accuracy: 0.92333335
    outputs:
 [[0.5758252  0.21203583 0.21213894]
 [0.21153729 0.21618372 0.572279  ]
 [0.21206658 0.5759078  0.2120256 ]
 ...
 [0.57577205 0.2122439  0.21198402]
 [0.2119722  0.21210259 0.5759252 ]
 [0.21234417 0.21194164 0.5757142 ]]

'''

# example 29 #########################################################


# example 30 #########################################################
