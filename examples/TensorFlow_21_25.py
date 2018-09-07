'''
21:lstm rnn.
22:自编码器，autoencoder。
23:
24:
25:
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
    除此以外的搭配方式都会报错ValueError: Shape must be rank 2 but is rank 1 for 'decoder_layer1_18/MatMul' (op: 'MatMul') with input shapes: [4], [4,8].
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


# example 24 #########################################################


# example 25 #########################################################


# example 26 #########################################################
