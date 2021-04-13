import numpy as np
import numpy.ma as npm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

train_data = np.load('./data_hw2/train_data.npy')
train_label = np.load('./data_hw2/train_label.npy')
test_data = np.load('./data_hw2/test_data.npy')
test_label = np.load('./data_hw2/test_label.npy')

scaler = StandardScaler()
scaler.fit(train_data)
Train_data = scaler.transform(train_data)
Test_data = scaler.transform(test_data)

a = np.where(train_label==1)
b = np.where(train_label==0)
c = np.where(train_label==-1)
A = Train_data[a[0]]
B = Train_data[b[0]]
C = Train_data[c[0]]

len_a = int(len(A)/2)
len_b = int(len(B)/2)
len_c = int(len(C)/2)

A_1 = A[0:len_a]
A_2 = A[len_a:]
B_1 = B[0:len_b]
B_2 = B[len_b:]
C_1 = C[0:len_c]
C_2 = C[len_c:]


def train(A_1,A_2,B_1,B_2):
    data_1 = np.vstack((A_1,B_1))
    data_2 = np.vstack((A_1,B_2))
    data_3 = np.vstack((A_2,B_1))
    data_4 = np.vstack((A_2,B_2))

    label_a_1 = np.ones(len(A_1))
    label_a_2 = np.ones(len(A_2))
    label_b_1 = np.ones(len(B_1))*-1
    label_b_2 = np.ones(len(B_2))*-1

    label_1 = np.hstack((label_a_1,label_b_1))
    label_2 = np.hstack((label_a_1,label_b_2))
    label_3 = np.hstack((label_a_2,label_b_1))
    label_4 = np.hstack((label_a_2,label_b_2))

    A1B1 = SVC(kernel = 'linear',gamma='auto',cache_size=2000 ,C=0.03)
    A1B2 = SVC(kernel = 'linear',gamma='auto',cache_size=2000 ,C=0.03)
    A2B1 = SVC(kernel = 'linear',gamma='auto',cache_size=2000 ,C=0.03)
    A2B2 = SVC(kernel = 'linear',gamma='auto',cache_size=2000 ,C=0.03)

    A1B1.fit(data_1,label_1)
    AB_1 = A1B1.decision_function(Test_data)

    A1B2.fit(data_2,label_2)
    AB_2 = A1B2.decision_function(Test_data)

    A2B1.fit(data_3,label_3)
    AB_3 = A2B1.decision_function(Test_data)

    A2B2.fit(data_4,label_4)
    AB_4 = A2B2.decision_function(Test_data)

    return np.maximum(np.minimum(AB_1,AB_2),np.minimum(AB_3,AB_4))

#train 3 classifiers some can be reused
class_1 = train(A_1,A_2,B_1,B_2)
class_2 = train(A_1,A_2,C_1,C_2)
class_3 = train(B_1,B_2,C_1,C_2)

a_label = np.minimum(class_1,class_2)
b_label = np.minimum(-class_1,class_3)
c_label = np.minimum(-class_2,-class_3)

Final_tag = []
Final_class = np.maximum(a_label,np.maximum(b_label,c_label))
for i in range(len(class_1)):
    if(Final_class[i] == a_label[i]):
        Final_tag.append(1)
    elif Final_class[i] == b_label[i]:
        Final_tag.append(0)
    else:
        Final_tag.append(-1)
        
test_label = np.load('./data_hw2/test_label.npy')

print(confusion_matrix(test_label,Final_tag))
