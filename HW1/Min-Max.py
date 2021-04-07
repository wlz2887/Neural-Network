#*-*coding:utf-8*-*
import numpy as np
import random
from random import random
from random import seed
import math
import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt
import time
%matplotlib widget

#input_layer:输入值
#hidden:隐藏层计算的结果
#network:权重
#output:输出层计算的结果
#true_val:真实值


data = np.loadtxt('../Desktop/神经网络/two-spiral traing data(update).txt') #change the path to your own path
x = data[:,0:2]
y = data[:,-1]


X,Y=[],[]
Black = np.arange(0,194,2)
np.random.shuffle(Black)
White = np.arange(1,194,2)
np.random.shuffle(White)#随机切分

a,b = x[Black],x[White]
# a = a[np.argsort(a[:,0]),:] 基于经验做纵向切割
# b = b[np.argsort(b[:,0]),:]
W1 = b[0:52]
W2 = b[45:97]
B1 = a[0:52]
B2 = a[45:97]

#生成训四个子集
X.append(np.vstack((W1,B1)))
Y.append(np.hstack((y[White][0:52],y[Black][0:52])))

X.append(np.vstack((W1,B2)))
Y.append(np.hstack((y[White][0:52],y[Black][45:97])))

X.append(np.vstack((W2,B1)))
Y.append(np.hstack((y[White][45:97],y[Black][0:52])))

X.append(np.vstack((W2,B2)))
Y.append(np.hstack((y[White][45:97],y[Black][45:97])))

#sigmoid函数
def sigmoid(x):
    return 1.0 / (1.0+np.exp(-x))


def sigmoidDeriviation(y):
    return np.multiply(y,1-y)


#初始化权重，输入参数为每层的神经元数量，输出的network记录的是结点间的权重
def init(n_input,n_hidden,n_output):
#     input_layer = [random() for i in range(n_input)]
    output = np.zeros(n_output) #1*1
    hidden = np.zeros(n_hidden) #(n_hidden,)
    
    v1 = np.random.randn(n_hidden,n_input)
    v2 = np.random.randn(n_output,n_hidden)
    u1 = np.random.randn(n_hidden,n_input)
    u2 = np.random.randn(n_output,n_hidden)
    return v1,v2,u1,u2,hidden,output


# seed(4)
n_input,n_hidden,n_output = 2,40,1
v1,v2,u1,u2,hidden,output = init(n_input,n_hidden,n_output)
Bias = [random() for i in range(2)]


Rate = [0.03,0.1,0.3]
Epoch = 10000
loss = []
data = []

for experiment in range(0,4):
    rate = 0.03
    start = time.perf_counter()
    for times in range(0,Epoch):
        Loss=0
        for i in range(0,104): #on-line learning
            input_layer=X[experiment][i]
            true_val=Y[experiment][i]
            hidden = np.dot(v1,input_layer)+np.dot(u1,input_layer**2)+Bias[0]
            hidden = sigmoid(hidden)#计算隐藏层
            hidden = hidden.reshape(n_hidden,1)
            output = sigmoid(np.dot(v2,hidden)+np.dot(u2,hidden**2)+Bias[1])#计算输出层，结束前向计算

            #预处理
            input_layer = input_layer.reshape(n_input,1)
            deriv1 = sigmoidDeriviation(hidden) #hidden(1-hidden)
            deriv2 = sigmoidDeriviation(output) #output(1-output)
            delta2 = -np.multiply((true_val-output),deriv2)#(e*out*(1-out))
            coeff = np.dot(delta2.T,v2)+np.dot(delta2.T,np.multiply(u2.T,hidden).T)#输出层产生的梯度

            #反向传播
            tmp3 = np.multiply(coeff.T,deriv1)
            v1 -= rate*(np.dot(tmp3,input_layer.T)) 
            u1 -= rate*(np.dot(tmp3,(input_layer**2).T))

            tmp1 = np.dot(delta2,hidden.T)
            tmp2 = np.dot(delta2,(hidden**2).T)
            v2 -= rate*tmp1
            u2 -= rate*tmp2

            #记录loss变化
            Loss += 0.5*(true_val-output)**2 
            if(i==193): #you can change the interval of recording loss as you wish
                loss.append(Loss[0]/194)
    end = time.perf_counter()
    print(end-start)

    
    Res=300
    Output = np.zeros(Res)
    a=np.linspace(-6,6,Res)
    for i in a:
        b = np.ones(Res)*i
        c=np.vstack((a,b))
        input_layer = c
        hidden = np.dot(v1,input_layer)+np.dot(u1,input_layer**2)+Bias[0]
        hidden = sigmoid(hidden)
        # hidden = hidden.reshape(40,1)
        # input_layer = input_layer.reshape(2,1)
        output = sigmoid(np.dot(v2,hidden)+np.dot(u2,hidden**2)+Bias[1])#向前传播结束
    #     print(output)
        Output=np.vstack((Output,output))

    data.append(Output[1:,])
  
  
#子图结果
lim=6
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.imshow(data[0],cmap='Greys',extent=[-lim,lim,-lim,lim])
ax2 = fig.add_subplot(222)
ax2.imshow(data[1],cmap='Greys',extent=[-lim,lim,-lim,lim])
ax3 = fig.add_subplot(223)
ax3.imshow(data[2],cmap='Greys',extent=[-lim,lim,-lim,lim])
ax4 = fig.add_subplot(224)
ax4.imshow(data[3],cmap='Greys',extent=[-lim,lim,-lim,lim])
plt.show()


#Min-Max后结果
Final = np.maximum(np.minimum(data[0],data[2]),np.minimum(data[1],data[3]))
lim=6
fig = plt.figure()
plt.imshow(Final,cmap='Greys',extent=[-lim,lim,-lim,lim])
plt.show()
