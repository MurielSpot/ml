'''
1:plot函数绘制函数线条。
2:imshow绘制图像，28x28。
3:
'''

# example 1 ########################################################################
# 画了三个图形。
import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(0,50,20)
y=np.linspace(0,50,20)

x2=np.random.normal(0,10,[20])
y2=np.random.normal(0,10,[20])

plt.show()
plt.plot(x,y,x2,y2,x+10,y)
# x,y画出一条线；
# x2,y2画出折线，但仍然是连在一起的；
# x+10,y画出一条线，即x,y对应的线向左平移了10个单位。

# example 2 ########################################################################
# imshow绘制图像，28x28。
import numpy as np
import matplotlib.pyplot as plt

datasets_path=r""#npy格式的图像文件夹路径。

def read_data(folder_path):
    plt.show()
    plt.ion()#交互式显示，即图像窗口内图像可以变化。

    for file in os.listdir(folder_path):
        file_full_path=os.path.join(folder_path,file)
        a_npy_data=np.load(file_full_path,"r")

        #print(type(a_npy_data))
        for img in a_npy_data[:1]:
            #要先将784个像素形状改变成28x28之后，才能传给imshow，imshow中的shape不指定为28x28也能正确显示图像。
            plt.imshow(np.reshape(img,(28,28)),shape=(28,28))
            #暂停一段时间，否则图像一闪而过，就看不到图像了。
            plt.pause(1)
        
read_data(datasets_path) 

# example 3 ########################################################################


# example 4 ########################################################################


# example 5 ########################################################################


# example 6 ########################################################################


# example 7 ########################################################################
