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


#AB classifier
data_1 = np.vstack((A_1,B_1))
data_2 = np.vstack((A_1,B_2))
data_3 = np.vstack((A_2,B_1))
data_4 = np.vstack((A_2,B_2))

label_a_1 = np.ones(len_a)
label_a_2 = np.ones(len(A)-len_a)
label_b_1 = np.ones(len_b)*-1
label_c_1 = np.ones(len_c)*-1
label_b_2 = np.ones(len(B)-len_b)*-1
label_c_2 = np.ones(len(C)-len_c)*-1

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

Final_AB = np.maximum(np.minimum(AB_1,AB_2),np.minimum(AB_3,AB_4))

#AC classifier
data_1 = np.vstack((A_1,C_1))
data_2 = np.vstack((A_1,C_2))
data_3 = np.vstack((A_2,C_1))
data_4 = np.vstack((A_2,C_2))

label_a_1 = np.ones(len_a)
label_a_2 = np.ones(len(A)-len_a)
label_b_1 = np.ones(len_b)*-1
label_c_1 = np.ones(len_c)*-1
label_b_2 = np.ones(len(B)-len_b)*-1
label_c_2 = np.ones(len(C)-len_c)*-1

label_1 = np.hstack((label_a_1,label_c_1))
label_2 = np.hstack((label_a_1,label_c_2))
label_3 = np.hstack((label_a_2,label_c_1))
label_4 = np.hstack((label_a_2,label_c_2))

C1A1 = SVC(kernel='linear',gamma='auto',cache_size=2000 ,C=0.03)
C1A2 = SVC(kernel='linear',gamma='auto',cache_size=2000 ,C=0.03)
C2A1 = SVC(kernel='linear',gamma='auto',cache_size=2000 ,C=0.03)
C2A2 = SVC(kernel='linear',gamma='auto',cache_size=2000 ,C=0.03)

C1A1.fit(data_1,label_1)
AC_1 = C1A1.predict(Test_data)
C2A1.fit(data_2,label_2)
AC_2 = C2A1.predict(Test_data)
C1A2.fit(data_3,label_3)
AC_3 = C1A2.predict(Test_data)
C2A2.fit(data_4,label_4)
AC_4 = C2A2.predict(Test_data)
Final_AC = np.maximum(np.minimum(AC_1,AC_2),np.minimum(AC_3,AC_4))

#BC classifier
data_1 = np.vstack((B_1,C_1))
data_2 = np.vstack((B_1,C_2))
data_3 = np.vstack((B_2,C_1))
data_4 = np.vstack((B_2,C_2))

label_a_1 = np.ones(len_a)
label_a_2 = np.ones(len(A)-len_a)
label_b_1 = np.ones(len_b)
label_c_1 = np.ones(len_c)*-1
label_b_2 = np.ones(len(B)-len_b)
label_c_2 = np.ones(len(C)-len_c)*-1

label_1 = np.hstack((label_b_1,label_c_1))
label_2 = np.hstack((label_b_1,label_c_2))
label_3 = np.hstack((label_b_2,label_c_1))
label_4 = np.hstack((label_b_2,label_c_2))

B1C1 = SVC(kernel='linear',gamma='auto',cache_size=2000 ,C=0.03)
B1C2 = SVC(kernel='linear',gamma='auto',cache_size=2000 ,C=0.03)
B2C1 = SVC(kernel='linear',gamma='auto',cache_size=2000 ,C=0.03)
B2C2 = SVC(kernel='linear',gamma='auto',cache_size=2000 ,C=0.03)

B1C1.fit(data_1,label_1)
res_1 = B1C1.predict(Test_data)
B1C2.fit(data_2,label_2)
res_2 = B1C2.predict(Test_data)
B2C1.fit(data_3,label_3)
res_3 = B2C1.predict(Test_data)
B2C2.fit(data_4,label_4)
res_4 = B2C2.predict(Test_data)
Final_BC = np.maximum(np.minimum(res_1,res_2),np.minimum(res_3,res_4))

#vote max
tmp_label = np.load('./data_hw2/test_label.npy')
Final_tag = []
cnt_a,cnt_b,cnt_c = np.zeros(len(Final_AB)),np.zeros(len(Final_AB)),np.zeros(len(Final_AB))
for i in range(len(Final_AB)):
    if(A_class[i]==1):
        cnt_a[i]+=1
    else:
        cnt_b[i]+=1
        cnt_c[i]+=1
    if(B_class[i]==1):
        cnt_b[i]+=1
    else:
        cnt_a[i]+=1
        cnt_c[i]+=1
    if(C_class[i]==1):
        cnt_c[i]+=1
    else:
        cnt_a[i]+=1
        cnt_b[i]+=1
    
vote_max = np.maximum(cnt_a,np.maximum(cnt_b,cnt_c))
for i in range(len(Final_AB)):
    if(cnt_c[i]==vote_max[i]):
        Final_tag.append(-1)
    elif (vote_max[i]==cnt_b[i]):
        Final_tag.append(0)
    else:
        Final_tag.append(1)
print(confusion_matrix(tmp_label,Final_tag))
