import torch
import torch.nn as nn
import torch.nn.functional as F


class model(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,num_layers):
        super(model,self).__init__()
        self.input_dim = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.hidden2out = nn.Linear(hidden_size,3)
#         self.hidden = self.init_hidden()
        self.prob = nn.LogSoftmax(dim=1)

    def  init_hidden(self,n):
        return (torch.randn(self.num_layers,n,self.hidden_size).double(),
                torch.randn(self.num_layers,n,self.hidden_size).double())

    def forward(self,seq):
        lstm_out, self.hidden = self.lstm(
            seq
            ,self.hidden #hidden 是c0,h0
        )
#         print(lstm_out.shape)
        outdat = self.hidden2out(lstm_out.reshape(len(lstm_out)*43,64))
        outlabel = self.prob(outdat)
#         print(outlabel.shape)
        return outlabel
