#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:34:20 2017
This is a small project for CNN in KERAS.
This file creates, trains and save a convolutional neural network for
Human Acitivity Recognition. The data we used for this file is released and provided by
Wireless Sensor Data Mining (WISDM) lab and can be found on this link.
http://www.cis.fordham.edu/wisdm/dataset.php  
Feel free to use this code and site this repositry if you use it for your reports or project.
@author: Muhammad Shahnawaz
"""
# importing libraries and dependecies 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
#from keras import backend as K
from keras import optimizers
#K.set_image_dim_ordering('th')
# setting up a random seed for reproducibility
random_seed = 611
np.random.seed(random_seed)

#%%

# matplotlib inline
plt.style.use('ggplot')
# defining function for loading the dataset
def readData(filePath):
    # attributes of the dataset
    columnNames = ['user_id','activity','timestamp','x-axis','y-axis','z-axis']
    data = pd.read_csv(filePath,header = None, names=columnNames,na_values=';')
    return data
# defining a function for feature normalization
# (feature - mean)/stdiv
def featureNormalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset-mu)/sigma
# defining the function to plot a single axis data
def plotAxis(axis,x,y,title):
    axis.plot(x,y)
    axis.set_title(title)
    axis.xaxis.set_visible(False)
    axis.set_ylim([min(y)-np.std(y),max(y)+np.std(y)])
    axis.set_xlim([min(x),max(x)])
    axis.grid(True)
# defining a function to plot the data for a given activity
def plotActivity(activity,data):
    fig,(ax0,ax1,ax2) = plt.subplots(nrows=3, figsize=(15,10),sharex=True)
    plotAxis(ax0,data['timestamp'],data['x-axis'],'x-axis')
    plotAxis(ax1,data['timestamp'],data['y-axis'],'y-axis')
    plotAxis(ax2,data['timestamp'],data['z-axis'],'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.9)
    plt.show()
# defining a window function for segmentation purposes
def windows(data,size):
    start = 0
    while start< data.count():
        yield int(start), int(start + size)
        start+= (size/2)
# segmenting the time series
def segment_signal(data, window_size = 90):
    segments = np.empty((0,window_size,3))
    labels= np.empty((0))
    for (start, end) in windows(data['timestamp'],window_size):
        x = data['x-axis'][start:end]
        y = data['y-axis'][start:end]
        z = data['z-axis'][start:end]
        if(len(data['timestamp'][start:end])==window_size):
            segments = np.vstack([segments,np.dstack([x,y,z])])
            labels = np.append(labels,stats.mode(data['activity'][start:end])[0][0])
    return segments, labels
''' Main Code '''
# # # # # # # # #   reading the data   # # # # # # # # # # 
# Path of file #
dataset = readData('actitracker_raw.txt')
# plotting a subset of the data to visualize
for activity in np.unique(dataset['activity']):
    subset = dataset[dataset['activity']==activity][:180]
    plotActivity(activity,subset)
# segmenting the signal in overlapping windows of 90 samples with 50% overlap
segments, labels = segment_signal(dataset)
# Normalize the segments
segments = featureNormalize(segments)
#categorically defining the classes of the activities
labels = np.asarray(pd.get_dummies(labels),dtype = np.int8)
# defining parameters for the input and network layers
# we are treating each segmeent or chunk as a 2D image (90 X 3)
numOfRows = segments.shape[1]
numOfColumns = segments.shape[2]
numChannels = 1
numFilters = 128 # number of filters in Conv2D layer
# kernal size of the Conv2D layer
kernalSize1 = 2
# max pooling window size
poolingWindowSz = 2
# number of filters in fully connected layers
numNueronsFCL1 = 128
numNueronsFCL2 = 128
# split ratio for test and validation
trainSplitRatio = 0.8
# number of epochs
Epochs = 10
# batchsize
batchSize = 10
# number of total clases
numClasses = labels.shape[1]
# dropout ratio for dropout layer
dropOutRatio = 0.2
# reshaping the data for network input
reshapedSegments = segments.reshape(segments.shape[0], numOfRows, numOfColumns,1)
# splitting in training and testing data
trainSplit = np.random.rand(len(reshapedSegments)) < trainSplitRatio
trainX = reshapedSegments[trainSplit]
testX = reshapedSegments[~trainSplit]
trainX = np.nan_to_num(trainX)
testX = np.nan_to_num(testX)
trainY = labels[trainSplit]
testY = labels[~trainSplit]

print('numRows: ' + str(numOfRows))
print('numColumns: ' + str(numOfColumns))
print('numClasses: ' + str(numClasses))

#%%
# ---- Old Keras implementation ----

# def cnnModel():
#     model = Sequential()
#     # adding the first convolutionial layer with 32 filters and 5 by 5 kernal size, using the rectifier as the activation function
#     model.add(Conv2D(numFilters, (kernalSize1,kernalSize1),input_shape=(numOfRows, numOfColumns,1),activation='relu'))
#     # adding a maxpooling layer
#     model.add(MaxPooling2D(pool_size=(poolingWindowSz,poolingWindowSz),padding='valid'))
#     # adding a dropout layer for the regularization and avoiding over fitting
#     model.add(Dropout(dropOutRatio))
#     # flattening the output in order to apply the fully connected layer
#     model.add(Flatten())
#     # adding first fully connected layer with 256 outputs
#     model.add(Dense(numNueronsFCL1, activation='relu'))
#     #adding second fully connected layer 128 outputs
#     model.add(Dense(numNueronsFCL2, activation='relu'))
#     # adding softmax layer for the classification
#     model.add(Dense(numClasses, activation='softmax'))
#     # Compiling the model to generate a model
#     adam = optimizers.Adam(lr = 0.001, decay=1e-6)
#     model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#     return model
# model = cnnModel()
# for layer in model.layers:
#     print(layer.name)
# model.fit(trainX,trainY, validation_split=1-trainSplitRatio,epochs=10,batch_size=batchSize,verbose=2)
# score = model.evaluate(testX,testY,verbose=2)
# print('Baseline Error: %.2f%%' %(100-score[1]*100))
# model.save('model.h5')
np.save('groundTruth.npy',testY)
np.save('testData.npy',testX)
np.save('trainData.npy',trainX)
np.save('trainLabels.npy',trainY)

#%%

# ------------ Our own implementation of the model architecture in PyTorch --------------

from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

testX = np.load('testData.npy')
testY = np.load('groundTruth.npy')
trainX = np.load('trainData.npy')
trainY = np.load('trainLabels.npy')

# Convert the training and testing data to PyTorch tensors
trainX_torch = torch.from_numpy(np.transpose(trainX, (0, 3, 1, 2))).float()
trainY_torch = torch.from_numpy(np.argmax(trainY, axis=1)).long()
testX_torch = torch.from_numpy(np.transpose(testX, (0, 3, 1, 2))).float()
testY_torch = torch.from_numpy(np.argmax(testY, axis=1)).long()

# Create TensorDatasets for training and testing data
train_data = TensorDataset(trainX_torch, trainY_torch)
test_data = TensorDataset(testX_torch, testY_torch)

# Create DataLoaders for training and testing data
train_loader = DataLoader(train_data, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batchSize, shuffle=False)

class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.conv1 = nn.Conv2d(numChannels, numFilters, (kernalSize1, kernalSize1))
        self.pool = nn.MaxPool2d((poolingWindowSz, poolingWindowSz))
        self.dropout = nn.Dropout(dropOutRatio)
        self.fc1 = nn.Linear(numFilters * ((numOfRows - kernalSize1 + 1) // poolingWindowSz) * ((numOfColumns - kernalSize1 + 1) // poolingWindowSz), numNueronsFCL1)
        self.fc2 = nn.Linear(numNueronsFCL1, numNueronsFCL2)
        self.fc3 = nn.Linear(numNueronsFCL2, numClasses)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = x.view(-1, numFilters * ((numOfRows - kernalSize1 + 1) // poolingWindowSz) * ((numOfColumns - kernalSize1 + 1) // poolingWindowSz))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = PyTorchModel()

# Define the optimizer
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(Epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss = train_loss/len(train_loader.dataset)

    # Testing loop
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = test_loss/len(test_loader.dataset)
    test_accuracy = correct / total

    print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'model_pytorch.pth')
