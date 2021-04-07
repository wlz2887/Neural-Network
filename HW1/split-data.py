#如果是随机选取，则使用random.shuffle
Black = np.arange(0,194,2)
# np.random.shuffle(Black)
White = np.arange(1,194,2)
# np.random.shuffle(White)


#纵向对半分，加一点重叠部分
a,b = x[Black],x[White]
tmpa = a[np.argsort(a[:,0]),:]
tmpb = b[np.argsort(b[:,0]),:]
W1 = tmpb[0:52]
W2 = tmpb[45:97]
B1 = tmpa[0:52]
B2 = tmpa[45:97]


#训练四次记录四个权重，分别计算output再输入min_max modular
def validation():
    I=np.linspace(-6,6,100)
    for i in I:
        for j in I:
            input_layer=np.array([i,j])
            input_layer = input_layer.reshape(2,1)
            hidden_1 = sigmoid(np.dot(v1_1,input_layer)+np.dot(u1_1,input_layer**2)+Bias[0])
            hidden_1 = hidden_1.reshape(40,1)
            output_1 = sigmoid(np.dot(v2_1,hidden_1)+np.dot(u2_1,hidden_1**2)+Bias[1])#向前传播结束
            
            hidden_2 = sigmoid(np.dot(v1_2,input_layer)+np.dot(u1_2,input_layer**2)+Bias[0])
            hidden_2 = hidden_2.reshape(40,1)
            output_2 = sigmoid(np.dot(v2_2,hidden_2)+np.dot(u2_2,hidden_2**2)+Bias[1])#向前传播结束
            
            hidden_3 = sigmoid(np.dot(v1_3,input_layer)+np.dot(u1_3,input_layer**2)+Bias[0])
            hidden_3 = hidden_3.reshape(40,1)
            output_3 = sigmoid(np.dot(v2_3,hidden_3)+np.dot(u2_3,hidden_3**2)+Bias[1])#向前传播结束
            
            hidden_4 = sigmoid(np.dot(v1_4,input_layer)+np.dot(u1_4,input_layer**2)+Bias[0])
            hidden_4 = hidden_4.reshape(40,1)
            output_4 = sigmoid(np.dot(v2_4,hidden_4)+np.dot(u2_4,hidden_4**2)+Bias[1])#向前传播结束
            
            output = max(min(output_1,output_2),min(output_3,output_4))
            if(output>0.5):#0
                plt.scatter(i,j,c='orange',s=2)
            else:
                plt.scatter(i,j,c='blue',s=2)
validation()
