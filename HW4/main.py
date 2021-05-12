# -*- code=utf-8 -*-
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from lstm import model
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

train_data = np.load('./train_data.npy')
test_data = np.load('./test_data.npy')
train_label = np.load('./train_label.npy')
test_label = np.load('./test_label.npy')

#Normalization
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)


# train_label = torch.from_numpy(train_label+1).long()
# test_label = torch.from_numpy(test_label+1).long()
# train_data = torch.from_numpy(train_data)
# test_data = torch.from_numpy(test_data)

loader = []
loader2 = []
for i in range(0,11):
    a = train_data[i*3397:(i+1)*3397]
    b = train_label[i*3397:(i+1)*3397]+1
    dat,tag = list(),list()
    look_back = 43

    for _ in range(79):
        dat.append(a[_*look_back:_*look_back+look_back])
        tag.append(b[_*look_back:_*look_back+look_back])
    dat = torch.from_numpy(np.array(dat)).double()
    tag = torch.from_numpy(np.array(tag)).long()
#     train = TensorDataset(train_data[i*3397:(i+1)*3397],train_label[i*3397:(i+1)*3397])
    train = TensorDataset(dat,tag)
    train_loader = DataLoader(dataset = train, shuffle = False, batch_size = 1, num_workers = 8,) #将batch_size当做序列长度?
    loader.append(train_loader)

for i in range(0,4):
    a = test_data[i*3397:(i+1)*3397]
    b = test_label[i*3397:(i+1)*3397]+1
    dat,tag = list(),list()

    for _ in range(79):
        dat.append(a[_*43:_*43+43])

        dat.append(a[_*43:_*43+43])
        tag.append(b[_*43:_*43+43])
    dat = torch.from_numpy(np.array(dat)).double()
    tag = torch.from_numpy(np.array(tag)).long()
    test = TensorDataset(dat,tag)
    test_loader = DataLoader(dataset = test, shuffle = False, batch_size = 4,num_workers = 8)
    loader2.append(test_loader)

#============================================================================================
#train

batch_size = 4
net = model(310,64,batch_size,6) #seq_len and hidden_dim, batch_size is set to default 1
net = net.double()
optimizer = optim.Adam(net.parameters(),lr = 0.0001)
# loss_func = nn.CrossEntropyLoss()
loss_func = nn.NLLLoss()

for p in net.parameters():
    p.requires_grad = True

for epoch in range(100):
    for i in range(11):
        net.zero_grad()
        Loader = loader[i]
        loader_iter = iter(Loader)
        for j in range(len(Loader)):
            data,label = loader_iter.next()
            Batch_size = len(data)
            label = label.view(Batch_size*43,-1).squeeze(dim=1)
            net.hidden = net.init_hidden(Batch_size)
#             input_size = len(data)
            class_out = net(data)
            loss = loss_func(class_out,label)

            loss.backward()
            optimizer.step()
    print('epoch = ',epoch,' loss = ',loss)

    true_class = 0
    for i in range(11):
        Loader = loader[i]
        loader_iter = iter(Loader)
        for j in range(len(Loader)):
            data,label = loader_iter.next()
            Batch_size = len(data)
            label = label.view(Batch_size*43,-1).squeeze(dim=1)
            net.hidden = net.init_hidden(Batch_size)
            input_size = len(data)
            class_out = net(data)
            out_label = torch.argmax(class_out,dim=1)
            true_class += (out_label==label).sum().item()
    print('epoch = ',epoch,'acc on train = ', true_class/3397/11)
    
    true_class = 0
    for i in range(4):
        Loader = loader2[i]
        loader_iter = iter(Loader)
        for j in range(len(Loader)):
            data,label = loader_iter.next()
            Batch_size = len(data)
            label = label.view(43*Batch_size,-1).squeeze(dim=0)
            net.hidden = net.init_hidden(Batch_size)
            
            class_out = net(data)
            out_label = torch.argmax(class_out,dim=1)
            true_class += (out_label==label).sum().item()
    print('epoch = ',epoch,'acc on test = ',true_class/3397/4)

torch.save(net,'/lustre/home/acct-hpc/hpcwlz/neural-network/HW4/my_model')
                                                                              
