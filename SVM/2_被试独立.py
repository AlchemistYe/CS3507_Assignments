import os
import numpy as np  
from sklearn import svm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


def load_person(path):
    data = np.array([])
    label = np.array([])
    data_temp_1 = np.array([])
    label_temp_1 = np.array([])
    data_temp_2 = np.array([])
    label_temp_2 = np.array([])
    if(os.path.exists(os.path.join(path, 'train_data.npy'))):
            data_temp_1 = np.load(os.path.join(path, 'train_data.npy'))
            data_temp_1 = data_temp_1.reshape(data_temp_1.shape[0], (data_temp_1.shape[1] * data_temp_1.shape[2]))
            label_temp_1 = np.load(os.path.join(path, 'train_label.npy'))
    if(os.path.exists(os.path.join(path, 'test_data.npy'))):
            data_temp_2 = np.load(os.path.join(path, 'test_data.npy'))
            data_temp_2 = data_temp_2.reshape(data_temp_2.shape[0], (data_temp_2.shape[1] * data_temp_2.shape[2]))
            label_temp_2 = np.load(os.path.join(path, 'test_label.npy'))

    data = np.append(data_temp_1, data_temp_2, axis=0)
    label = np.append(label_temp_1, label_temp_2, axis=0)

    return data, label


def load_group(path):
    data = []
    label = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            data_temp, label_temp = load_person(os.path.join(root,d))
            if(len(data_temp) > 0):
                data.append(data_temp)
                label.append(label_temp)
    
    data_sub = np.array([])
    label_sub = np.array([])
    data_out = []
    label_out = []
    for i in range(15):
        #data_sub = np.append(data[i], data[i+15], axis=0)
        #data_sub = np.append(data_sub, data[i+30], axis=0)
        data_out.append(data[i+30])
        #label_sub = np.append(label[i], label[i+15], axis=0)
        #label_sub = np.append(label_sub, label[i+30], axis=0)
        label_out.append(label[i+30])

    return data_out, label_out


ori_path = "C:\\Users\\hp\\Downloads\\SEEDIV"
data_group, label_group = load_group(os.path.join(ori_path))


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
        #predictor = svm.SVC(gamma='scale', C=1.0, kernel='linear', max_iter=100000, probability=True)
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


def get_test_train(data_g, label_g, index):
    test_data = data_g[index]
    test_label = label_g[index]
    
    train_data = np.array([])
    train_label = np.array([])
     
    if(index == 0):
        train_data = data_g[1]
        train_label = label_g[1]
        for i in range(2,15):
            train_data = np.append(train_data, data_g[i], axis=0)
            train_label = np.append(train_label, label_g[i], axis=0) 
    else:
        train_data = data_g[0]
        train_label = label_g[0]
        for i in range(1,15):
            if(i != index):
                train_data = np.append(train_data, data_g[i], axis=0)
                train_label = np.append(train_label, label_g[i], axis=0) 

    return train_data, train_label, test_data, test_label

accuracy = 0.
for i in range(15):
    train_data, train_label, test_data, test_label = get_test_train(data_group, label_group, i)   
    accuracy += calc_accuracy(train_data, train_label, test_data, test_label)

print(accuracy / 15)








