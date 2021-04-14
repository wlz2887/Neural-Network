import numpy as np
import numpy.ma as npm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

#读取数据
train_data = np.load('./data_hw2/train_data.npy')
train_label = np.load('./data_hw2/train_label.npy')
test_data = np.load('./data_hw2/test_data.npy')
test_label = np.load('./data_hw2/test_label.npy')


#Normalization
scaler = StandardScaler()
tmp = np.vstack((train_data,test_data))
scaler.fit(tmp)
Train_data = scaler.transform(train_data)
Test_data = scaler.transform(test_data)


#找到标签所在的index
a = np.where(train_label==1)
b = np.where(train_label==0)
c = np.where(train_label==-1)
Label = train_label


#分别训练三个两分类器
Label[a] = 1
Label[b] = -1
Label[c] = -1
Model_a = SVC(gamma='auto',cache_size=5000,probability=False,C=0.03)
Model_a.fit(Train_data,Label)
res_a = Model_a.decision_function(Test_data)

Label[a] = -1
Label[b] = 1
Label[c] = -1
Model_b = SVC(gamma='auto',cache_size=5000,probability=False,C=0.03)
Model_b.fit(Train_data,Label)
res_b = Model_b.decision_function(Test_data)

Label[a] = -1
Label[b] = -1
Label[c] = 1
Model_c = SVC(gamma='auto',cache_size=5000,probability=False,C=0.03)
Model_c.fit(Train_data,Label)
res_c = Model_c.decision_function(Test_data)


#整合三个分类器的结果
pred = []
Final = np.maximum(res_a,np.maximum(res_b,res_c))
# Final = np.maximum(res_a,res_b,res_c)
for i in range(0,Final.size):
    if(Final[i]==res_a[i]):
        pred.append(1)
    elif(Final[i]==res_b[i]):
        pred.append(0)
    elif(Final[i]==res_c[i]):
        pred.append(-1)
print(confusion_matrix(test_label,pred))#打印最终分类器的混淆矩阵结果

        
#验证分类器在测试集上的准确率        
test_a = np.where(test_label==1)
test_b = np.where(test_label==0)
test_c = np.where(test_label==-1)
Label_a = test_label
Label_a[test_a] = 1
Label_a[test_b] = -1
Label_a[test_c] = -1
res_a = Model_a.score(Test_data,Label_a)
print(res_a)

Label_a = test_label
Label_a[test_a] = -1
Label_a[test_b] = 1
Label_a[test_c] = -1
res_b = Model_b.score(Test_data,Label_a)
print(res_b)

Label_a = test_label
Label_a[test_a] = -1
Label_a[test_b] = -1
Label_a[test_c] = 1
res_c = Model_c.score(Test_data,Label_a)
print(res_c)

