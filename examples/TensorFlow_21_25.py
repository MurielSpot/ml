'''
21:lstm rnn.
22:自编码器，autoencoder。
23:name_scope,tf.get_variable,tf.Variable的用法.
24:variable_scope,reuse,tf.get_variable.
25:batch normaliztion.
'''

# example 21 #########################################################
# lstm rnn.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_START=0
TIME_STEPS=20
BATCH_SIZE=50
INPUT_SIZE=1
OUTPUT_SIZE=1
CELL_SIZE=10
LR=0.006
BATCH_START_TEST=0

def get_batch():
    global BATCH_START,TIME_STEPS
    # xs shape (50 batch,20 steps)
    xs=np.arange(BATCH_START,BATCH_START+TIME_STEPS*BATCH_SIZE).reshape(BATCH_SIZE,TIME_STEPS)/(10*np.pi)
    seq=np.sin(xs)
    res=np.cos(xs)
    BATCH_START+=TIME_STEPS
    return [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]

class LSTMRNN(object):
    def __init__(self,n_steps,input_size,output_size,cell_size,batch_size):
        self.n_steps=n_steps
        self.input_size=input_size
        self.output_size=output_size
        self.cell_size=cell_size
        self.batch_size=batch_size
        with tf.name_scope("inputs"):
            self.xs=tf.placeholder(tf.float32,[None,n_steps,input_size],name="xs")
            self.ys=tf.placeholder(tf.float32,[None,n_steps,output_size],name="ys")
        with tf.variable_scope("in_hidden"):
            self.add_input_layer()
        with tf.variable_scope("LSTM_cell"):
            self.add_cell()
        with tf.variable_scope("out_hidden"):
            self.add_output_layer()
        with tf.name_scope("cost"):
            self.compute_cost()
        with tf.name_scope("train"):
            self.train_op=tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self):
        l_in_x=tf.reshape(self.xs,[-1,self.input_size],name="to_2D")#(batch*n_step,in_size).
        Ws_in=self._weight_variable([self.input_size,self.cell_size])
        bs_in=self._bias_variable([self.cell_size])
        with tf.name_scope("Wx_plus_b"):
            l_in_y=tf.matmul(l_in_x,Ws_in)+bs_in
        self.l_in_y=tf.reshape(l_in_y,[-1,self.n_steps,self.cell_size],name="to_3D")

    def add_cell(self):
        lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(self.cell_size,forget_bias=1.0,state_is_tuple=True)
        with tf.name_scope("initial_state"):
            self.cell_init_state=lstm_cell.zero_state(self.batch_size,dtype=tf.float32)
            self.cell_outputs,self.cell_final_state=tf.nn.dynamic_rnn(lstm_cell,self.l_in_y,initial_state=self.cell_init_state,time_major=False)


    def add_output_layer(self):
        l_out_x=tf.reshape(self.cell_outputs,[-1,self.cell_size],name="to_2D")
        Ws_out=self._weight_variable([self.cell_size,self.output_size])
        bs_out=self._bias_variable([self.output_size])
        with tf.name_scope("Wx_plus_b"):
            self.pred=tf.matmul(l_out_x,Ws_out)+bs_out

    def compute_cost(self):
        # 求每一步的loss。
        #losses=tf.nn.seq2seq.sequence_loss_by_example(
        losses=tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            logits=[tf.reshape(self.pred,[-1],name="reshape_pred")],
            targets=[tf.reshape(self.ys,[-1],name="reshape_target")],
            weights=[tf.ones([self.batch_size*self.n_steps],dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.msr_error,
            name="losses"
            )
        # 将整个TensorFlow的loss相加，再除以整个batch_size，即对loss做平均，得到对于某个batch的总cost。
        with tf.name_scope("average_cost"):
            self.cost=tf.div(tf.reduce_sum(losses,name="losses_sum"),tf.cast(self.batch_size,tf.float32),name="average_cost")
            tf.summary.scalar("cost",self.cost)

    def msr_error(self,labels,logits):
        return tf.square(tf.subtract(logits,labels))

    def _weight_variable(self,shape,name="weights"):
        return tf.get_variable(shape=shape,initializer=tf.random_normal_initializer(mean=0.,stddev=1.),name=name)

    def _bias_variable(self,shape,name="biases"):
        return tf.get_variable(shape=shape,initializer=tf.constant_initializer(0.1),name=name)

def main():
    model=LSTMRNN(TIME_STEPS,INPUT_SIZE,OUTPUT_SIZE,CELL_SIZE,BATCH_SIZE)
    with tf.Session() as sess:
        merged=tf.summary.merge_all()
        writer=tf.summary.FileWriter("logs",sess.graph)

        sess.run(tf.global_variables_initializer())

        plt.ion()#交互模式。
        plt.show()

        for i in range(200):
            # sequence,result,xs.
            seq,res,xs=get_batch()
            if i==0:
                feed_dict={
                    model.xs:seq,
                    model.ys:res
                    }
            else:
                feed_dict={
                    model.xs:seq,
                    model.ys:res,
                    model.cell_init_state:state # use last state as the initial state for this run.
                    }

            _,cost,state,pred=sess.run([model.train_op,model.cost,model.cell_final_state,model.pred],feed_dict=feed_dict
                )

            # plotting
            plt.plot(xs[0,:],res[0].flatten(),'r',xs[0,:],pred.flatten()[:TIME_STEPS],'b--')#result 用红色线，pred用蓝色虚线。
            plt.ylim((-1.2,1.2))
            plt.draw()
            plt.pause(0.3)

            if i%20==0:
                print("cost:",round(cost,4))
                result=sess.run(merged,feed_dict)
                writer.add_summary(result,i)
                
main()

# example 22 #########################################################
# 自编码器，autoencoder。
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

data_path=r"./ml_datasets/MNIST_data/"
mnist=input_data.read_data_sets(data_path,one_hot=False)

learning_rate=0.01
training_epoches=5
batch_size=256
display_step=1#每隔多少个epoch显示一次结果。
example_to_show=10

net_input=784

x=tf.placeholder(tf.float32,[None,net_input])

def init_weights(in_size,out_size):
    return tf.Variable(tf.random_normal([in_size,out_size],mean=0.0,stddev=1.0))

def init_biases(size):
    return  tf.Variable(tf.random_normal([size]))

hidden_neuron_num={
    "layer1":256,
    "layer2":128,
    "layer3":64,
    "layer4":32,
    "layer5":16,
    "layer6":8,
    "layer7":4
    }
def encoder(x):
    layers=[]

    # first.
    with tf.name_scope("encoder_layer1"):
        layer=tf.matmul(x,init_weights(net_input,hidden_neuron_num["layer1"]))+init_biases(hidden_neuron_num["layer1"])
        layer=tf.nn.sigmoid(layer)
        layers.append(layer)
    
    # the hidden layer after first layer of encoder.
    len_of_layers=len(hidden_neuron_num)
    # 层次：[2,7].
    for i in range(2,len_of_layers+1):
        with tf.name_scope("encoder_layer"+str(i)):
            W=init_weights(hidden_neuron_num["layer"+str(i-1)],hidden_neuron_num["layer"+str(i)])
            b=init_biases(hidden_neuron_num["layer"+str(i)])
            layer=tf.matmul(layers[-1],W)+b
            layer=tf.nn.sigmoid(layer)
            layers.append(layer)

    return layers[-1]#返回最后一层会报错.

def decoder(x):
    '''
    注意这里：
    decoder函数开头的layers=[]，layers.append(x)应该与encoder函数里的return layers[-1]搭配。
    或者decoder函数开头的layers=[]，layers.append(x)改为layers=x,此时应该与encoder函数里的return layers搭配。
    除此以外的搭配方式都会报错
        ValueError: Shape must be rank 2 but is rank 1 for 'decoder_layer1_18/MatMul' (op: 'MatMul') with input shapes: [4], [4,8].
    
    出错原因举个例子：
    def a():
    list=[(2,3),(2,3)]
    return list[-1]

    def b(x):
        list=x
        print(list[-1])

        list=[]
        list.append(x)
        print(list[-1])

    b(a())
    
    结果：
    3
    (2, 3)
    
    所以自编码器的程序中，如果encoder返回的是最后一个tensor，直接用x=tensor，再取x[-1]得到的将是tensor里的一个部分，
    而x=[tensor]，再取x[-1]得到的才是整个tensor。
    '''
    layers=[]
    layers.append(x)

    len_of_layers=len(hidden_neuron_num)

    # the hidden layer of decoder.
    # i范围[1,7).
    for i in range(1,len_of_layers):
        # decoder_layer范围[1,6].
        with tf.name_scope("decoder_layer"+str(i)):
            '''
            # 从6到1，7-i
            "layer1":256,
            "layer2":128,
            "layer3":64,
            "layer4":32,
            "layer5":16,
            "layer6":8,
            "layer7":4
            '''
            # 7-1=6 => len_of_layers-i=encoder_hidden_layer
            encoder_hidden_layer=len_of_layers-i
            W=init_weights(hidden_neuron_num["layer"+str(encoder_hidden_layer+1)],hidden_neuron_num["layer"+str(encoder_hidden_layer)])
            b=init_biases(hidden_neuron_num["layer"+str(encoder_hidden_layer)])
            layer=tf.matmul(layers[-1],W)+b
            layer=tf.nn.sigmoid(layer)
            layers.append(layer)

    # last layer.
    with tf.name_scope("decoder_layer"+str(len_of_layers)):
        W=init_weights(hidden_neuron_num["layer"+str(1)],net_input)
        b=init_biases(net_input)
        layer=tf.matmul(layers[-1],W)+b
        layer=tf.nn.sigmoid(layer)
        layers.append(layer)

    return layers[-1]

encoder_op=encoder(x)
decoder_op=decoder(encoder_op)

# loss
loss=tf.reduce_mean(tf.pow(x-decoder_op,2))

# optimizer
train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    total_batch=int(mnist.train.num_examples/batch_size)

    for epoch in range(training_epoches):
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            _train_op,_loss=sess.run([train_op,loss],feed_dict={x:batch_xs})
            
        if epoch%display_step==0:
            print("epoch [%d]"%epoch," loss:",_loss)            

    print("train end\nnow to visualize\n")

    encoder_decoder=sess.run(decoder_op,feed_dict={x:mnist.test.images[:example_to_show]})

    # compare original images with their reconstructions.
    f,a=plt.subplots(2,10,figsize=(10,2))
    for i in range(example_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[1][i].imshow(np.reshape(encoder_decoder[i],(28,28)))
    plt.show()

'''
程序运行结果：ml/examples/TensorFlow_22_autoencoder.PNG
epoch [0]  loss: 0.11168995
epoch [1]  loss: 0.10286156
epoch [2]  loss: 0.09679584
epoch [3]  loss: 0.09334956
epoch [4]  loss: 0.09041786
''' 

# example 23 #########################################################
# name_scope,tf.get_variable,tf.Variable的用法.
import tensorflow as tf

with tf.name_scope("a_name_scope"):
    initializer=tf.constant_initializer(value=1)
    # 使用get_variable建立变量打印的时候，不会包含scope名。
    var1=tf.get_variable(name="var1",shape=[1],dtype=tf.float32)
    # 使用variable建立变量时，打印时会包含scope名。
    var2=tf.Variable(name="var2",initial_value=[2],dtype=tf.float32)
    # 使用variable建立变量时，虽然name="var2"，但是打印出来的名字会在var2后面添加序号。
    var21=tf.Variable(name="var2",initial_value=[2.1],dtype=tf.float32)
    var22=tf.Variable(name="var2",initial_value=[2.2],dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(var1.name,sess.run(var1))
    print(var2.name,sess.run(var2))
    print(var21.name,sess.run(var21))
    print(var22.name,sess.run(var22))
    
'''
结果：
var1:0 [-1.3536651]
a_name_scope/var2:0 [2.]
a_name_scope/var2_1:0 [2.1]
a_name_scope/var2_2:0 [2.2]
'''

# example 24 #########################################################
# variable_scope,reuse,tf.get_variable.
import tensorflow as tf

with tf.variable_scope("a_variable_scope",reuse=True):
    initializer=tf.constant_initializer(value=3)
    # 得到var3变量。
    var3=tf.get_variable(name="var3",shape=[1],dtype=tf.float32,initializer=initializer)
    # 允许reuse之后，可以重新使用这个var3，即下面这个var3_reuse和上面的var3是同一个变量。
    var3_reuse=tf.get_variable(name="var3")

    # 下面两句看起来是重用，但实际在TensorFlow里得到了两个不同的变量。
    var4=tf.Variable(name="var4",initial_value=[4],dtype=tf.float32)
    var4_reuse=tf.Variable(name="var4",initial_value=[4.1],dtype=tf.float32)
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(var3.name,sess.run(var3))
    print(var3_reuse.name,sess.run(var3_reuse))
    print(var4.name,sess.run(var4))
    print(var4_reuse.name,sess.run(var4_reuse))

'''
结果：
a_variable_scope/var3:0 [3.]
a_variable_scope/var3:0 [3.]
a_variable_scope_1/var4:0 [4.]
a_variable_scope_1/var4_1:0 [4.1]
'''

# example 25 #########################################################
# 批正则化，别人写的代码，后面画图的部分还没看完。
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# Hyper parameters
N_SAMPLES = 2000
BATCH_SIZE = 64
EPOCH = 12
LR = 0.03
N_HIDDEN = 8
ACTIVATION = tf.nn.tanh
B_INIT = tf.constant_initializer(-0.2)      # use a bad bias initialization

# training data
# 在-7到10中，选取N_SAMPLES个值，添加新维度后，形状变为[N_SAMPLES,1]，即形成一列。
x = np.linspace(-7, 10, N_SAMPLES)[:, np.newaxis]
# 打乱产生的数，多维数组打乱第一维。
np.random.shuffle(x)
# 从均值0，标准差2的分布里选取值，填到x.shape形状的的数组里，得到noise。
noise = np.random.normal(0, 2, x.shape)
# 已经有x轴左边，再生成随机数据的y轴坐标。
y = np.square(x) - 5 + noise
# 水平方向堆叠数组，相当于按第二维拼接数组。
# 即[1,2],[2,3]按第二维拼接得到[1,2,2,3].
train_data = np.hstack((x, y))

# test data
test_x = np.linspace(-7, 10, 200)[:, np.newaxis]
noise = np.random.normal(0, 2, test_x.shape)
test_y = np.square(test_x) - 5 + noise

# plot input data
# 绘制散点图。
plt.scatter(x, y, c='#FF9359', s=50, alpha=0.5, label='train')
#  用来显示多个图例。
plt.legend(loc='upper left')

# tensorflow placeholder
tf_x = tf.placeholder(tf.float32, [None, 1])
tf_y = tf.placeholder(tf.float32, [None, 1])
tf_is_train = tf.placeholder(tf.bool, None)     # flag for using BN on training or testing

class NN(object):
    def __init__(self, batch_normalization=False):
        self.is_bn = batch_normalization

        self.w_init = tf.random_normal_initializer(0., .1)  # weights initialization
        self.pre_activation = [tf_x]
        if self.is_bn:
            self.layer_input = [tf.layers.batch_normalization(tf_x, training=tf_is_train)]  # for input data
        else:
            self.layer_input = [tf_x]
        for i in range(N_HIDDEN):  # adding hidden layers
            self.layer_input.append(self.add_layer(self.layer_input[-1], 10, ac=ACTIVATION))
        '''
       全连接层：dense(
                    inputs,#inputs: 输入数据，2维tensor. 
                    units,#units: 该层的神经单元结点数。
                    activation=None,#activation: 激活函数.
                    use_bias=True,#use_bias: Boolean型，是否使用偏置项. 
                    kernel_initializer=None,#kernel_initializer: 卷积核的初始化器. 
                    bias_initializer=tf.zeros_initializer(),#bias_initializer: 偏置项的初始化器，默认初始化为0. 
                    kernel_regularizer=None,#kernel_regularizer: 卷积核化的正则化，可选. 
                    bias_regularizer=None,#bias_regularizer: 偏置项的正则化，可选. 
                    activity_regularizer=None,#activity_regularizer: 输出的正则化函数. 
                    trainable=True,#trainable: Boolean型，表明该层的参数是否参与训练。如果为真则变量加入到图集合中 GraphKeys.TRAINABLE_VARIABLES (see tf.Variable). 
                    name=None,#name: 层的名字. 
                    reuse=None#reuse: Boolean型, 是否重复使用参数.
                )
        '''
        self.out = tf.layers.dense(self.layer_input[-1], 1, kernel_initializer=self.w_init, bias_initializer=B_INIT)
        self.loss = tf.losses.mean_squared_error(tf_y, self.out)

        # !! IMPORTANT !! the moving_mean and moving_variance need to be updated,
        # pass the update_ops with control_dependencies to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train = tf.train.AdamOptimizer(LR).minimize(self.loss)

    def add_layer(self, x, out_size, ac=None):
        x = tf.layers.dense(x, out_size, kernel_initializer=self.w_init, bias_initializer=B_INIT)
        self.pre_activation.append(x)
        # the momentum plays important rule. the default 0.99 is too high in this case!
        if self.is_bn: x = tf.layers.batch_normalization(x, momentum=0.4, training=tf_is_train)    # when have BN
        out = x if ac is None else ac(x)
        return out

nets = [NN(batch_normalization=False), NN(batch_normalization=True)]    # two nets, with and without BN

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# plot layer input distribution
f, axs = plt.subplots(4, N_HIDDEN+1, figsize=(10, 5))
plt.ion()   # something about plotting

def plot_histogram(l_in, l_in_bn, pre_ac, pre_ac_bn):
    for i, (ax_pa, ax_pa_bn, ax,  ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):
        [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]
        if i == 0: p_range = (-7, 10); the_range = (-7, 10)
        else: p_range = (-4, 4); the_range = (-1, 1)
        ax_pa.set_title('L' + str(i))
        ax_pa.hist(pre_ac[i].ravel(), bins=10, range=p_range, color='#FF9359', alpha=0.5)
        ax_pa_bn.hist(pre_ac_bn[i].ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5)
        ax.hist(l_in[i].ravel(), bins=10, range=the_range, color='#FF9359')
        ax_bn.hist(l_in_bn[i].ravel(), bins=10, range=the_range, color='#74BCFF')
        for a in [ax_pa, ax, ax_pa_bn, ax_bn]:
            a.set_yticks(()); a.set_xticks(())
        ax_pa_bn.set_xticks(p_range); ax_bn.set_xticks(the_range); axs[2, 0].set_ylabel('Act'); axs[3, 0].set_ylabel('BN Act')
    plt.pause(0.01)

losses = [[], []]   # record test loss
for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    np.random.shuffle(train_data)
    step = 0
    in_epoch = True
    while in_epoch:
        b_s, b_f = (step*BATCH_SIZE) % len(train_data), ((step+1)*BATCH_SIZE) % len(train_data) # batch index
        step += 1
        if b_f < b_s:
            b_f = len(train_data)
            in_epoch = False
        b_x, b_y = train_data[b_s: b_f, 0:1], train_data[b_s: b_f, 1:2]         # batch training data
        sess.run([nets[0].train, nets[1].train], {tf_x: b_x, tf_y: b_y, tf_is_train: True})     # train

        if step == 1:
            l0, l1, l_in, l_in_bn, pa, pa_bn = sess.run(
                [nets[0].loss, nets[1].loss, nets[0].layer_input, nets[1].layer_input,
                 nets[0].pre_activation, nets[1].pre_activation],
                {tf_x: test_x, tf_y: test_y, tf_is_train: False})
            [loss.append(l) for loss, l in zip(losses, [l0, l1])]   # recode test loss
            plot_histogram(l_in, l_in_bn, pa, pa_bn)     # plot histogram

plt.ioff()

# plot test loss
plt.figure(2)
plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
plt.plot(losses[1], c='#74BCFF', lw=3, label='Batch Normalization')
plt.ylabel('test loss'); plt.ylim((0, 2000)); plt.legend(loc='best')

# plot prediction line
pred, pred_bn = sess.run([nets[0].out, nets[1].out], {tf_x: test_x, tf_is_train: False})
plt.figure(3)
plt.plot(test_x, pred, c='#FF9359', lw=4, label='Original')
plt.plot(test_x, pred_bn, c='#74BCFF', lw=4, label='Batch Normalization')
plt.scatter(x[:200], y[:200], c='r', s=50, alpha=0.2, label='train')
plt.legend(loc='best'); plt.show()
