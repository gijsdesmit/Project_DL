import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# DON"T SET RANDOM SEED BECAUSE THE DEFENSE MUST BE RANDOM

## Define variables

numChannels = 1
numFilters = 128  # number of filters in Conv2D layer
kernalSize1 = 2  # kernal size of the Conv2D layer
poolingWindowSz = 2
numNueronsFCL1 = 128
numNueronsFCL2 = 128
dropOutRatio = 0.2
numOfRows = 90
numOfColumns = 3
numClasses = 6
batchSize = 10
Epochs = 10


class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.conv1 = nn.Conv2d(numChannels, numFilters, (kernalSize1, kernalSize1))
        self.pool = nn.MaxPool2d((poolingWindowSz, poolingWindowSz))
        self.dropout = nn.Dropout(dropOutRatio)
        self.fc1 = nn.Linear(numFilters * ((numOfRows - kernalSize1 + 1) // poolingWindowSz) * (
                    (numOfColumns - kernalSize1 + 1) // poolingWindowSz), numNueronsFCL1)
        self.fc2 = nn.Linear(numNueronsFCL1, numNueronsFCL2)
        self.fc3 = nn.Linear(numNueronsFCL2, numClasses)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = x.view(-1, numFilters * ((numOfRows - kernalSize1 + 1) // poolingWindowSz) * (
                    (numOfColumns - kernalSize1 + 1) // poolingWindowSz))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# These define the new input dimensions

#TODO: perhaps add more ratios; (6,180), (7,210)
n_pc = 150  # number of padded columns # increase way more, try 110x10
n_pr = 5  # number of padded rows, make 5?
ratios = [(3,90), (4,120), (5,150)]

# ratio = ratios[np.random.randint(0, len(ratios))]
# print(ratio)

# For implementation 1 and 2
#n_pc = 95
#n_pr = 5

class PyTorchModel_defended(nn.Module):
    def __init__(self):
        super(PyTorchModel_defended, self).__init__()
        self.conv1 = nn.Conv2d(numChannels, numFilters, (kernalSize1, kernalSize1))
        self.pool = nn.MaxPool2d((poolingWindowSz, poolingWindowSz))
        self.dropout = nn.Dropout(dropOutRatio)
        self.fc1 = nn.Linear(numFilters * ((n_pr - kernalSize1 + 1) // poolingWindowSz) * ((n_pc - kernalSize1 + 1) // poolingWindowSz), numNueronsFCL1)
        self.fc2 = nn.Linear(numNueronsFCL1, numNueronsFCL2)
        self.fc3 = nn.Linear(numNueronsFCL2, numClasses)

    def forward(self, x):
        # x has shape: (batchSize, numChannels, numOfRows or timesteps, numOfColumns)

        # Implementation 1:
        # (batchsize, 1, 90, 3)
        #padded_x = torch.zeros((x.shape[0], x.shape[1], n_pc, n_pr))
        #height_offset = np.random.randint(0, n_pc - x.shape[2])
        #width_offset = np.random.randint(0, n_pr - x.shape[3])
        ##remove this line? height_offset, width_offset = np.random.randint(0, 3, size=2)
        #padded_x[:, :, height_offset:height_offset+90, width_offset:width_offset+3] = x
        #x = padded_x

        # Implementation 2:
        #height_offset = np.random.randint(0, n_pc - x.shape[2])
        #width_offset = np.random.randint(0, n_pr - x.shape[3])

        # Use torch.nn.functional.pad for efficient padding
        #x = F.pad(x, (width_offset, n_pr - width_offset - x.shape[3], height_offset, n_pc - height_offset - x.shape[2]))

        # Implementation 3:
        ratio = ratios[np.random.randint(0, len(ratios))]
        w = ratio[0]
        h = ratio[1]
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        # if the ratio is not the same as the last one, then we need to pad to fit n_pc and n_pr
        if (w != n_pr) and (h != n_pc):
            h_offset = np.random.randint(0, n_pc - x.shape[2])
            w_offset = np.random.randint(0, n_pr - x.shape[3])
            x = F.pad(x, (w_offset, n_pr - w_offset - x.shape[3], h_offset, n_pc - h_offset - x.shape[2]))

            # remove line? #  x = F.pad(x, (0, n_pr - w, 0, n_pc - h))
            # x = F.pad(x, (w, n_pr - w - x.shape[3], h, n_pc - h - x.shape[2]))

        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = x.view(-1, numFilters * ((n_pr - kernalSize1 + 1) // poolingWindowSz) * ((n_pc - kernalSize1 + 1) // poolingWindowSz))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#%%
