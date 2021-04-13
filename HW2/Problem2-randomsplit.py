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
# tmp = np.vstack((train_data,test_data))
scaler.fit(train_data)
Train_data = scaler.transform(train_data)
Test_data = scaler.transform(test_data)

train_label = np.load('./data_hw2/train_label.npy')
a = np.where(train_label==1)
nota = np.where(train_label!=1)
np.random.shuffle(nota)
b = np.where(train_label==0)
notb = np.where(train_label!=0)
np.random.shuffle(notb)
c = np.where(train_label==-1)
notc = np.where(train_label!=-1)
np.random.shuffle(notc)
A = Train_data[a[0]]
B = Train_data[b[0]]
C = Train_data[c[0]]
notA = Train_data[nota[0]]
notB = Train_data[notb[0]]
notC = Train_data[notc[0]]

#切分另一半数据
def build_not(notA,len_nota):
    notA_1 = notA[0:len_nota]
    notA_2 = notA[len_nota:len_nota*2]
    notA_3 = notA[len_nota*2:len_nota*3]
    notA_4 = notA[len_nota*3:]
    return notA_1,notA_2,notA_3,notA_4
  
len_a = int(len(A)/2)
len_b = int(len(B)/2)
len_c = int(len(C)/2)
len_nota = int(len(notA)/4)
len_notb = int(len(notB)/4)
len_notc = int(len(notC)/4)

A_1 = A[0:len_a]
A_2 = A[len_a:]
B_1 = B[0:len_b]
B_2 = B[len_b:]
C_1 = C[0:len_c]
C_2 = C[len_c:]

notA_1,notA_2,notA_3,notA_4 = build_not(notA,len_nota)
notB_1,notB_2,notB_3,notB_4 = build_not(notB,len_notb)
notC_1,notC_2,notC_3,notC_4 = build_not(notC,len_notc)

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
    AB_1 = A1B1.predict(Test_data)

    A1B2.fit(data_2,label_2)
    AB_2 = A1B2.predict(Test_data)

    A2B1.fit(data_3,label_3)
    AB_3 = A2B1.predict(Test_data)

    A2B2.fit(data_4,label_4)
    AB_4 = A2B2.predict(Test_data)

    return np.maximum(np.minimum(AB_1,AB_2),np.minimum(AB_3,AB_4))
  
class_a_1 = train(A_1,A_2,notA_1,notA_2)
class_a_2 = train(A_1,A_2,notA_3,notA_4)
class_b_1 = train(B_1,B_2,notB_1,notB_2)
class_b_2 = train(B_1,B_2,notB_3,notB_4)
class_c_1 = train(C_1,C_2,notC_1,notC_2)
class_c_2 = train(C_1,C_2,notC_3,notC_4)

a_label = np.minimum(class_a_1,class_a_2)
b_label = np.minimum(class_b_1,class_b_2)
c_label = np.minimum(class_c_1,class_c_2)

Final_tag = []
Final_class = np.maximum(a_label,np.maximum(b_label,c_label))
for i in range(len(class_a_1)):
    if(Final_class[i] == a_label[i]):
        Final_tag.append(1)
    elif Final_class[i] == b_label[i]:
        Final_tag.append(0)
    else:
        Final_tag.append(-1)
        
test_label = np.load('./data_hw2/test_label.npy')

print(confusion_matrix(test_label,Final_tag))
