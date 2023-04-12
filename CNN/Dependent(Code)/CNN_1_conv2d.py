import os
import torch
import numpy as np  
import torch.nn.functional as F

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
    label_com = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            data_temp, label_temp = load_one(os.path.join(root,d), set)
            if(data_temp.size > 0):
                #data_temp = data_temp.reshape(-1, data_temp.shape[1] * data_temp.shape[2])
                data_temp = torch.from_numpy(data_temp).type(torch.FloatTensor)
                data_temp = torch.unsqueeze(data_temp, 1)
                #data_temp = F.normalize(data_temp, dim=0)
                #data_temp = data_temp.reshape(-1, 1, data_temp.shape[1])
                #label_temp = np.reshape(label_temp, (-1, 1, 1))      
                #label_temp = torch.from_numpy(label_temp).type(torch.FloatTensor)
                label_temp_1 = np.zeros([len(label_temp), 4])
                for i in range(len(label_temp)):
                    label_temp_1[i][label_temp[i]] = 1.0    
                label_temp_1 = torch.from_numpy(label_temp_1).type(torch.FloatTensor)
                label_temp = torch.from_numpy(label_temp).type(torch.FloatTensor)

                data.append(data_temp)
                label.append(label_temp_1)
                label_com.append(label_temp)
    return data, label, label_com


ori_path = "C:\\Users\\hp\\Downloads\\SEEDIV"
set_1 = 'train'
set_2 = 'test'
train_data_all, train_label_all, train_label_com = load_all(ori_path, set_1)
test_data_all, test_label_all, test_label_com = load_all(ori_path, set_2)

print(train_data_all[0].shape)

import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import math

class CNN_net(nn.Module):
    def __init__(self):
        super(CNN_net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=2, padding=1),   # out (, )
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=2, padding=1),   # out (8, )
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=272, out_features=256),
            #nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            #nn.Dropout(0.6),
            nn.Linear(128, 4),
        )
        
     
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fc(out)
        return out

device = 'cuda'

learning_rate = 1e-3
epoches = 500

loss_train_all = 0.
acc__train_all = 0.
loss_test_all = 0.
acc_test_all = 0.
for time in range(len(train_data_all)):
    batch_size = math.floor(len(train_data_all[time]) / 10)

    criterion = nn.CrossEntropyLoss()
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
            x = train_data_all[time][start : end].to(device)
            y = train_label_all[time][start : end].to(device)
            #print(y.shape)
            y_pred = net(x).to(device)
            opt.zero_grad()
            y_pred = torch.squeeze(y_pred)

            loss = criterion(y_pred, y)
            loss.backward()
            opt.step()
            
        
            running_loss += loss.item()
            _, predict = torch.max(y_pred, 1)
            #print(predict.shape)
            #print(train_label_com[0][start : end].shape)
            correct_num = (predict.cpu() == train_label_com[time][start : end]).sum()
            running_acc += correct_num.item()

        running_loss /= len(train_data_all[1])
        running_acc /= len(train_data_all[1])
    
    print("[%d] [Train] Loss: %.5f, ACC: %.2f" %(time, running_loss, running_acc))
    loss_train_all += running_loss
    acc__train_all += running_acc

    net.eval()

    running_loss = 0.
    running_acc = 0.

    x = test_data_all[time].to(device)
    y = test_label_all[time].to(device)
    y_pred = net(x).to(device)
    y_pred = torch.squeeze(y_pred)
    running_loss += loss.item()
    _, predict = torch.max(y_pred, 1)
    predict = predict.cpu()
    #predict_0 = predict[np.where(predict == 0)]
    #predict_1 = predict[np.where(predict == 1)]
    #predict_2 = predict[np.where(predict == 2)]
    #predict_3 = predict[np.where(predict == 3)]
    #print("[%d] [0]:%d [1]:%d [2]:%d [3]:%d" %(time, len(predict_0), len(predict_1), len(predict_2), len(predict_3)))
    #print(train_label_com[0][start : end].shape)
    correct_num = (predict == test_label_com[time]).sum()
    running_acc += correct_num.item()

    running_loss /= len(test_data_all[time])
    running_acc /= len(test_data_all[time])
    print("[%d] [Test] Loss: %.5f, ACC: %.2f" %(time, running_loss, running_acc))
    loss_test_all += running_loss
    acc_test_all += running_acc

loss_train_all /= len(train_data_all)
acc__train_all /= len(train_data_all)
loss_test_all /= len(train_data_all)
acc_test_all /= len(train_data_all)

print("[Average] [Train] Loss: %.5f, ACC: %.2f" %(loss_train_all, acc__train_all))
print("[Average] [Test] Loss: %.5f, ACC: %.2f" %(loss_test_all, acc_test_all))
