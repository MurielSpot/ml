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


# example 4 #########################################################


# example 5 #########################################################
