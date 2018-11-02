# 这个网络准确率震荡得非常厉害，还不清楚为什么。里面用到的有些文件我还没上传。

# encoding:utf-8
import numpy as np

import os
from os.path import isfile, join

import re

# 使用IMDB情感分析数据集，IMDB训练集和测试集分别包含了25000条已标注的电影评价，
# 评价满分10分，负面评价小于等于4，正面大于等于7.

# 使用已经训练好的词典向量模型，该矩阵包含了40 0000 的文本向量，每行有50维的数据。

# [b'0' b',' b'.' ... b'rolonda' b'zsombor' b'unk']
wordsList = np.load('wordsList.npy')
print('载入word列表')

'''
numpy.ndarray.tolist
ndarray.tolist()
Return the array as a (possibly nested) list.
The array may be recreated, a = np.array(a.tolist()).
'''
wordsList=wordsList.tolist()

'''
str->bytes:encode编码
bytes->str:decode解码
'''
wordsList = [word.decode('UTF-8')
                for word in wordsList]

'''
加print报错，上网查到原因如下：

    print(b'\xc2\xbb'.decode('utf-8'))
    报错:UnicodeEncodeError: 'gbk' codec can't encode character '\xbb' in position 0: illegal multibyte sequence
    上网找了下utf-8编码表，发现的确特殊字符»的utf-8形式就是c2bb,unicode是'\u00bb'，为什么无法解码呢。。。
    仔细看看错误信息，它提示'gbk'无法encode，但是我的代码是utf-8无法decode，压根牛头不对马嘴，终于让我怀疑是print函数出错了。。于是立即有了以下的测试
    print('\u00bb')
    结果报错了：UnicodeEncodeError: 'gbk' codec can't encode character '\xbb' in position 0: illegal multibyte sequence
    原来是print()函数自身有限制，不能完全打印所有的unicode字符。
    知道原因后，google了一下解决方法，其实print()函数的局限就是Python默认编码的局限，因为系统是win7的，python的默认编码不是'utf-8',改一下python的默认编码成'utf-8'就行了
'''
#print(wordsList)

'''
[[ 0.         0.         0.        ...  0.         0.         0.       ]
    [ 0.013441   0.23682   -0.16899   ... -0.56657    0.044691   0.30392  ]
    [ 0.15164    0.30177   -0.16763   ... -0.35652    0.016413   0.10216  ]
    ...
    [-0.51181    0.058706   1.0913    ... -0.25003   -1.125      1.5863   ]
    [-0.75898   -0.47426    0.4737    ...  0.78954   -0.014116   0.6448   ]
    [-0.79149    0.86617    0.11998   ... -0.29996   -0.0063003  0.3954   ]]
'''
wordVectors = np.load('wordVectors.npy')
print('载入文本向量')

# 400000
print(len(wordsList))
# (400000, 50)
print(wordVectors.shape)

'''
#这部分跑一次之后也不用跑了
#在构造整个训练集索引之前，需要先可视化和分析数据的情况从而确定并设置最好的序列长度。
# 下面是预处理过程。
pos_files=['pos/'+f for f in os.listdir('pos/') if isfile(join('pos/',f))]
neg_files=['neg/'+f for f in os.listdir('neg/') if isfile(join('neg/',f))]

num_words=[]
for pf in pos_files:
    with open(pf,"r",encoding='utf-8') as f:
        line=f.readline()
        counter=len(line.split())
        num_words.append(counter)
print("读取正面评价结束")

for nf in neg_files:
    with open(nf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        num_words.append(counter)
print('负面评价完结')

np.save('num_words', np.array(num_words))
'''
num_words=np.load('num_words.npy')
print('正面负面评价训练数据读取完毕')

num_files = len(num_words)
print('文件总数', num_files)
print('所有的词的数量', sum(num_words))
print('平均文件词的长度', sum(num_words) / len(num_words))

# 可视化,发现大部分文本长度都在两百多字。
#import matplotlib
#import matplotlib.pyplot
##matplotlib.use('qt4agg')#??
## 指定字体
#matplotlib.rcParams['font.sans-serif']=['SimHei']
#matplotlib.rcParams['font.family']='sans-serif'
##% matplotlib inline #??
#matplotlib.pyplot.hist(num_words,50,facecolor='g')#Plot a histogram.
#matplotlib.pyplot.xlabel("文本长度")
#matplotlib.pyplot.ylabel("频次")
#matplotlib.pyplot.axis([0,1200,0,8000])
#matplotlib.pyplot.show()

# Dimensions for each word vector
num_dimensions = 50
# 每个文件序列长度人为设定为250.
max_seq_num=250

'''
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string=string.lower().replace("<br />"," ")
'''
    #re.sub(pattern, repl, string, count=0, flags=0)
    #pattern：表示正则表达式中的模式字符串；
    #repl：被替换的字符串（既可以是字符串，也可以是函数）；
    #string：要被处理的，要被替换的字符串；
    #count：匹配的次数, 默认是全部替换
'''
    return re.sub(strip_special_chars,"",string.lower())

# 下面这部分跑一次就可以了。

ids=np.zeros((num_files, max_seq_num), dtype='int32')
file_count = 0
for pf in pos_files:
    with open(pf,"r",encoding='utf-8') as f:
        print(file_count)
        indexCounter = 0
        line = f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()#默认分隔符为空白字符。
        for word in split:
            try:
                # index函数在字符串里找是否有word，如果找到则返回其索引，否则抛出一个异常。
                # 将word索引，填入ids矩阵里。
                ids[file_count][indexCounter]=wordsList.index(word)
            except ValueError:
                ids[file_count][indexCounter] = 399999  # 未知的词
            indexCounter = indexCounter + 1
            if indexCounter >= max_seq_num:
                break
        file_count+=1

for nf in neg_files:
    with open(nf, "r",encoding='utf-8') as f:
        print(file_count)
        indexCounter = 0
        line = f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                ids[file_count][indexCounter] = wordsList.index(word)
            except ValueError:
                ids[file_count][indexCounter] = 399999  # 未知的词语
            indexCounter = indexCounter + 1
            if indexCounter >= max_seq_num:
                break
        file_count = file_count + 1

np.save('idsMatrix', ids)
'''

from random import randint

batch_size=24
lstm_units=80
num_labels=2
iterations=100
lr=0.0001
# 一行存一句话，每个单元存单词对应的index。
ids = np.load('idsMatrix.npy')

def get_train_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(1, 11499)#pos
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)#neg
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels

def get_test_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels

import tensorflow as tf

tf.reset_default_graph()

labels=tf.placeholder(tf.float32,[batch_size,num_labels])
input_data=tf.placeholder(tf.int32,[batch_size,max_seq_num])
data=tf.Variable(tf.zeros([batch_size,max_seq_num,num_dimensions]),dtype=tf.float32)
data=tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell=tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstmCell=tf.contrib.rnn.DropoutWrapper(cell=lstmCell,output_keep_prob=0.5)
value,_=tf.nn.dynamic_rnn(lstmCell,data,dtype=tf.float32)

weight=tf.Variable(tf.truncated_normal([lstm_units,num_labels]))
bias=tf.Variable(tf.constant(0.1,shape=[num_labels]))
value=tf.transpose(value,[1,0,2])
last=tf.gather(value,int(value.get_shape()[0])-1)
prediction=(tf.matmul(last,weight)+bias)

correct_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(labels,1))
accuracy= tf.reduce_mean(tf.cast(correct_pred, tf.float32))

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=labels))
optimizer=tf.train.AdamOptimizer(lr).minimize(loss)

saver=tf.train.Saver()

with tf.Session() as sess:
    if os.path.exists("model") and os.path.exists("model/checkpoint"):
        saver.restore(sess, tf.train.latest_checkpoint('models'))
    else:
        sess.run(tf.global_variables_initializer())

    for step in range(1000):
        next_batch,next_batch_labels=get_test_batch()
        if step%5==0:
            print("step:", step, " 正确率:", (sess.run(
                accuracy, {input_data: next_batch, labels: next_batch_labels})) * 100)

    if not os.path.exists("models"):
        os.mkdir("models")
    save_path = saver.save(sess, "models/model.ckpt")
    #print("Model saved in path: %s" % save_path)
