import os
import torch
import numpy as np  

 
def load_person(path):
    data_temp_1 = np.array([])
    label_temp_1 = np.array([])
    data_temp_2 = np.array([])
    label_temp_2 = np.array([])
    label_temp_com1 = np.array([])
    label_temp_com2 = np.array([])
    if(os.path.exists(os.path.join(path, 'train_data.npy'))):
            data_temp_1 = np.load(os.path.join(path, 'train_data.npy'))
            #data_temp_1 = data_temp_1.swapaxes(1,2)

            label_temp_1 = np.load(os.path.join(path, 'train_label.npy'))
            label_temp_com1 = np.zeros([len(label_temp_1), 4])
            for i in range(len(label_temp_1)):
                label_temp_com1[i][label_temp_1[i]] = 1.0   

    if(os.path.exists(os.path.join(path, 'test_data.npy'))):
            data_temp_2 = np.load(os.path.join(path, 'test_data.npy'))
            #data_temp_2 = data_temp_2.swapaxes(1,2)

            label_temp_2 = np.load(os.path.join(path, 'test_label.npy'))
            label_temp_com2 = np.zeros([len(label_temp_2), 4])
            for i in range(len(label_temp_2)):
                label_temp_com2[i][label_temp_2[i]] = 1.0   

    data = np.append(data_temp_1, data_temp_2, 0)
    label = np.append(label_temp_com1, label_temp_com2, 0)
    label_com = np.append(label_temp_1, label_temp_2, 0)

    return data, label, label_com


def load_group(path):
    data = []
    label = []
    label_com = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            data_temp, label_temp, label_temp_com = load_person(os.path.join(root,d))
            if(len(data_temp) > 0):
                data_temp = torch.from_numpy(data_temp).type(torch.FloatTensor)
                label_temp = torch.from_numpy(label_temp).type(torch.FloatTensor)
                label_temp_com = torch.from_numpy(label_temp_com).type(torch.FloatTensor)
                data_temp = torch.unsqueeze(data_temp, 1)
                data.append(data_temp)
                label.append(label_temp)
                label_com.append(label_temp_com)
    
    #data_sub = np.array([])
    #label_sub = np.array([])
    data_out = []
    label_out = []
    label_com_out = []
    for i in range(15):
        #data_sub = np.append(data[i], data[i+15], axis=0)
        #data_sub = np.append(data_sub, data[i+30], axis=0)
        data_out.append(data[i+30])
        #label_sub = np.append(label[i], label[i+15], axis=0)
        #label_sub = np.append(label_sub, label[i+30], axis=0)
        label_out.append(label[i+30])
        label_com_out.append(label_com[i+30 ])    

    return data_out, label_out, label_com_out


ori_path = "C:\\Users\\hp\\Downloads\\SEEDIV"
data_group, label_group, label_com_group = load_group(os.path.join(ori_path))



def get_test_train(data_g, label_g, label_com_g, index):
    test_data = data_g[index]
    test_label = label_g[index]
    test_label_com = label_com_g[index]
    
    train_data = np.array([])
    train_label = np.array([])
    train_label_com = np.array([])
     
    if(index == 0):
        train_data = data_g[1]
        train_label = label_g[1]
        train_label_com = label_com_g[1]
        for i in range(2,15):
            train_data = np.append(train_data, data_g[i], axis=0)
            train_label = np.append(train_label, label_g[i], axis=0) 
            train_label_com = np.append(train_label_com, label_com_g[i], axis=0)
    else:
        train_data = data_g[0]
        train_label = label_g[0]
        train_label_com = label_com_g[0]
        for i in range(1,15):
            if(i != index):
                train_data = np.append(train_data, data_g[i], axis=0)
                train_label = np.append(train_label, label_g[i], axis=0) 
                train_label_com = np.append(train_label_com, label_com_g[i], axis=0)
    
    train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label).type(torch.FloatTensor)
    train_label_com = torch.from_numpy(train_label_com).type(torch.FloatTensor)

    return train_data, train_label, train_label_com, test_data, test_label, test_label_com



import torch
import torch.nn as nn
import torch.nn.functional as F
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
            #nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            #nn.Dropout(0.2),
            nn.Linear(128, 4),
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fc(out)
        return out

device = 'cuda'


def calc_accuracy(train_data, train_label, train_label_com, test_data, test_label, test_label_com):
    learning_rate = 1e-3
    epoches = 500

    loss_train_all = 0.
    acc__train_all = 0.
    loss_test_all = 0.
    acc_test_all = 0.
    batch_size = math.floor(len(train_data) / 100)

    criterion = nn.CrossEntropyLoss()
    torch.cuda.empty_cache()
    net = CNN_net().to(device)
    opt = optim.Adam(net.parameters(), lr=learning_rate)
    net.train()

    for epoch in range(epoches):
        running_loss = 0.
        running_acc = 0.
        for i in range(100):
            start = i * batch_size
            end = (i+1) * batch_size
            x = train_data[start : end].to(device)
            y = train_label[start : end].to(device)
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
            correct_num = (predict.cpu() == train_label_com[start : end]).sum()
            running_acc += correct_num.item()

        running_loss /= len(train_data)
        running_acc /= len(train_data)
    
    loss_train_all = running_loss
    acc__train_all = running_acc

    net.eval()

    running_loss = 0.
    running_acc = 0.

    x = test_data.to(device)
    y = test_label.to(device)
    y_pred = net(x).to(device)
    y_pred = torch.squeeze(y_pred)
    running_loss += loss.item()
    _, predict = torch.max(y_pred, 1)
    predict = predict.cpu()
    
    #predict_0 = predict[np.where(predict == 0)]
    #predict_1 = predict[np.where(predict == 1)]
    #predict_2 = predict[np.where(predict == 2)]
    #predict_3 = predict[np.where(predict == 3)]
    #print("[0]:%d [1]:%d [2]:%d [3]:%d" %(len(predict_0), len(predict_1), len(predict_2), len(predict_3)))
    
    #print(train_label_com[0][start : end].shape)
    correct_num = (predict == test_label_com).sum()
    running_acc += correct_num.item()

    running_loss /= len(test_data)
    running_acc /= len(test_data)

    loss_test_all = running_loss
    acc_test_all = running_acc

    return loss_train_all, acc__train_all, loss_test_all, acc_test_all


acc_train = 0.
acc_test = 0.
for i in range(len(data_group)):
    train_data, train_label, train_label_com, test_data, test_label, test_label_com = get_test_train(data_group, label_group, label_com_group, i)   
    loss_train_all, acc__train_all, loss_test_all, acc_test_all = calc_accuracy(train_data, train_label, train_label_com, test_data, test_label, test_label_com)
    print("[%d] [Train] Loss: %.5f, ACC: %.2f" %(i, loss_train_all, acc__train_all))
    print("[%d] [Test] Loss: %.5f, ACC: %.2f \n" %(i, loss_test_all, acc_test_all))
    acc_train += acc__train_all
    acc_test += acc_test_all

print(acc_train/len(data_group))
print(acc_test/len(data_group))
