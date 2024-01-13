# ------------ Our own implementation of the model architecture in PyTorch --------------

from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from model import PyTorchModel, PyTorchModel_defended

#%%

# DON'T SET RANDOM SEED BECAUSE THE DEFENSE MUST BE RANDOM, but maybe here doesn't matter?
# random_seed = 611
# np.random.seed(random_seed)

## Define variables

numChannels = 1
numFilters = 128 # number of filters in Conv2D layer
kernalSize1 = 2 # kernal size of the Conv2D layer
poolingWindowSz = 2
numNueronsFCL1 = 128
numNueronsFCL2 = 128
dropOutRatio = 0.2
numOfRows = 90
numOfColumns = 3
numClasses = 6
batchSize = 10
Epochs = 10

## Load data

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

#%%

#Comment/uncomment to load chosen model

#model = PyTorchModel()
model = PyTorchModel_defended()

#load_model_file = 'model_pytorch.pth'
#load_model_file = 'model_pytorch_defended.pth'
load_model_file = 'model_pytorch_defended_i3.pth'
save_model_file = load_model_file # Change to save model to different file

model.load_state_dict(torch.load(load_model_file)) # uncomment to load model and train it further

# Define the optimizer
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Training loop
print('Starting training...')
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
    torch.save(model.state_dict(), 'model_temp.pth') # Save temporary model

#%%
# Save the trained model, comment/uncomment
#torch.save(model.state_dict(), save_model_file)
#print(f'Saved model to {save_model_file}.')

#%%
