# 每个example可以单独运行。

# example 1 #############################################################
import numpy as np

# 查看numpy版本和配置信息
np.__version__  
np.__config__.show()

# 创建数组
# 我们可以通过给array函数传递Python的序列对象创建数组，如果传递的是多层嵌套的序列，将创建多维数组(下例中的变量c):
a = np.array([1, 2, 3, 4])
b = np.array((5, 6, 7, 8))
c = np.array([[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10]])
print(a)
print(b)
print(c)
print(c.dtype)

# 数组的大小
print(a.shape)
print(c.shape)

# 改变数组形状
c.shape=4,3
print(c)

# 当某个轴的元素为-1时，将根据数组元素的个数自动计算此轴的长度.
c.shape=2,-1
print(c)

# reshape方法，可以创建一个改变了尺寸的新数组，原数组的shape保持不变：
d=c.reshape((-1,2))
print(d)

# 但reshape之后，数组c和d其实共享数据存储内存区域，因此修改其中任意一个数组的元素都会同时修改另外一个数组的内容：
c[1][3]=100
print(c)
print(d)

# 可以通过dtype参数在创建时指定元素类型:
print(np.array((2),dtype=np.float))

# example 2 #############################################################
import numpy as np

# 先创建一个Python序列，然后通过array函数将其转换为数组，这样做显然效率不高。
# 因此NumPy提供了很多专门用来创建数组的函数。

# arange函数类似于python的range函数，通过指定开始值、终值和步长来创建一维数组，注意数组不包括终值:
print(np.arange(0,1,0.1))

# linspace函数通过指定开始值、终值和元素个数来创建一维数组，可以通过endpoint关键字指定是否包括终值，缺省设置是包括终值:
print(np.linspace(0, 1, 3))

# logspace函数和linspace类似，不过它创建等比数列，下面的例子产生1(10^0)到100(10^2)、有20个元素的等比数列:
print(np.logspace(0, 2, 20))

# 使用frombuffer, fromstring, fromfile等函数可以从字节序列创建数组.

# Python的字符串实际上是字节序列，每个字符占一个字节，
# 因此如果从字符串s创建一个8bit的整数数组的话，所得到的数组正好就是字符串中每个字符的ASCII编码:
s = "abcdefgh"
print(np.fromstring(s, dtype=np.int8))

# 如果从字符串s创建16bit的整数数组，那么两个相邻的字节就表示一个整数，
# 把字节98和字节97当作一个16位的整数，它的值就是98*256+97 = 25185。
# 可以看出内存中是以little endian(低位字节在前)方式保存数据的。
print(np.fromstring(s, dtype=np.int16))

# 如果把整个字符串转换为一个64位的双精度浮点数数组，那么它的值是:
print("np.fromstring(s, dtype=np.float)")

# example 3 #############################################################



# example 4 #############################################################

# example 5 #############################################################


# example 6 #############################################################
