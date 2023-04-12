# %%
import os
import torch
import numpy as np  


def load_one(path, set):
    data = np.array([])
    label = np.array([])
    if(set == 'train'):
        if(os.path.exists(os.path.join(path, 'train_data.npy'))):
            data = np.load(os.path.join(path, 'train_data.npy'))
            label = np.load(os.path.join(path, 'train_label.npy'))
    elif(set == 'test'):
        if(os.path.exists(os.path.join(path, 'test_data.npy'))):
            data = np.load(os.path.join(path, 'test_data.npy'))
            label = np.load(os.path.join(path, 'test_label.npy'))
    else: 
        print("input error")

    return data, label


def load_all(path, set):
    data = []
    label = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            data_temp, label_temp = load_one(os.path.join(root,d), set)
            if(data_temp.size > 0):
                data_temp = data_temp.reshape(-1, 1, data_temp.shape[1] * data_temp.shape[2])
                #data_temp = data_temp.swapaxes(1,2)    
                data_temp = torch.from_numpy(data_temp).type(torch.FloatTensor)
                label_temp = np.reshape(label_temp, (-1, 1, 1))      
                label_temp = torch.from_numpy(label_temp).type(torch.FloatTensor)

                data.append(data_temp)
                label.append(label_temp)
    return data, label


ori_path = "C:\\Users\\hp\\Downloads\\SEEDIV"
set_1 = 'train'
set_2 = 'test'
train_data_all, train_label_all = load_all(ori_path, set_1)
test_data_all, test_label_all = load_all(ori_path, set_2)


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt


class CNN_net(nn.Module):
    def __init__(self):
        super(CNN_net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3),
            #nn.BatchNorm1d(8),
            nn.Softmax(),
            nn.MaxPool1d(kernel_size=2),   # out (8, 154)
            #nn.Dropout(0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2, padding=1),
            #nn.BatchNorm1d(16),
            nn.Softmax(),
            nn.MaxPool1d(kernel_size=5),   # out (16, 31)
            #nn.Dropout(0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=2, padding=1),
            #nn.BatchNorm1d(1),
            nn.Softmax(),
            nn.MaxPool1d(kernel_size=30),   # out (1, 1)
            #nn.Dropout(0.3),
        )

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.5)
        self.fc_1 = nn.Linear(in_features=1, out_features=4)
        #self.fc_2 = nn.Linear(in_features=4, out_features=4)

     
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc_1(out)
        #out = self.dropout(out)
        #out = self.fc_2(out)
        out = self.softmax(out)
        return out




# %%
device = 'cuda'

learning_rate = 1e-4
epoches = 8000
batch_size = 61

criterion = nn.MSELoss()
torch.cuda.empty_cache()
net = CNN_net().to(device)
opt = optim.Adam(net.parameters(), lr=learning_rate)
net.train()

for epoch in range(epoches):
    running_loss = 0.
    running_acc = 0.
    for i in range(10):
        start = i * batch_size
        end = (i+1) * batch_size
        x = train_data_all[0][start : end].to(device)
        y = train_label_all[0][start : end].to(device)
        y_pred = net(x).to(device)
        opt.zero_grad()

        loss = criterion(y_pred, y)
        loss.backward()
        opt.step()
        
        y_pred = torch.squeeze(y_pred)

        running_loss += loss.item()
        _, predict = torch.max(y_pred, 1)
        predict = predict.reshape(-1, 1, 1)
        #print(predict)
        #print(train_label_all[0][start : end].shape)
        correct_num = (predict.cpu() == train_label_all[0][start : end]).sum()
        running_acc += correct_num.item()
    if epoch == epoches-1:
        print(predict)
        
    running_loss /= len(train_data_all[0])
    running_acc /= len(train_data_all[0])
    print("[%d/%d] Loss: %.5f, ACC: %.2f" %(epoch+1, epoches, running_loss, running_acc))




