import os
import numpy as np  
from sklearn import svm
from sklearn.svm import SVR
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


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
                data_temp = data_temp.reshape(data_temp.shape[0], (data_temp.shape[1] * data_temp.shape[2]))
                data.append(data_temp)
                label.append(label_temp)
    return data, label


ori_path = "C:\\Users\\hp\\Downloads\\SEEDIV"
set_1 = 'train'
set_2 = 'test'
train_data_all, train_label_all = load_all(ori_path, set_1)
test_data_all, test_label_all = load_all(ori_path, set_2)

'''
ori_path = "C:\\Users\\hp\\Downloads\\SEEDIV\\1\\1_20160518"
set_1 = 'train'
set_2 = 'test'
train_data, train_label = load_one(ori_path, set_1)
test_data, test_label = load_one(ori_path, set_2)

train_data = train_data.reshape(train_data.shape[0], (train_data.shape[1] * train_data.shape[2]))
test_data = test_data.reshape(test_data.shape[0], (test_data.shape[1] * test_data.shape[2]))
'''

def calc_accuracy(train_data, train_label, test_data, test_label):
    train_label_sub = [[] for _ in range(4)]
    for i in range(0,4):
        for j in range (len(train_label)):
            if(train_label[j] == i):
                train_label_sub[i].append(1)
            else:
                train_label_sub[i].append(0)
        train_label_sub[i] = np.array(train_label_sub[i])
    train_label_sub = np.array(train_label_sub)


    proba = []
    for i in range(0,4):
        predictor = svm.SVC(gamma='scale', C=1.0, kernel='rbf', probability=True)
        #predictor = svm.SVC(gamma='scale', C=1.0, kernel='linear', max_iter=1000000, probability=True)
        predictor.fit(train_data, train_label_sub[i])
        proba.append(predictor.predict_proba(test_data)[:,1])
    proba = np.array(proba)


    accuracy = 0.
    for k in range(len(test_label)):
        max_proba = 0
        max_index = 0
        for i in range(0,4):
            if(max_proba <= proba[i][k]):
                max_proba = proba[i][k]
                max_index = i
        if(max_index == test_label[k]):
            accuracy +=1.0

    return accuracy/ len(test_label)

accuracy = 0.
for i in range(len(train_data_all)):
    accuracy += calc_accuracy(train_data_all[i], train_label_all[i], test_data_all[i], test_label_all[i])

print(accuracy / len(train_data_all))















