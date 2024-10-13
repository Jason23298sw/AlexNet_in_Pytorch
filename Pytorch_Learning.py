###### This script is intended to understand the structure of AlexNet by implementing a similar neural network model in PyTorch ######

import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

# Assume the dimension of the image is 256*256*3 -> 3 channels
data = torch.ones(size=(20,3,256,256))

# Define a custom neural network model by extending nn.Module
class Model(nn.Module):
    def __init__(self):
        super().__init__()


        # Stage 1 -> Pass one Convolutional layers then follow a Pooling layer twice
    
        # First Convolutional layer with 3 input channels, 96 output channels, kernel size 11, stride 4
        self.conv1 = nn.Conv2d(3,96,kernel_size=11,stride=4)

        # First max Pooling layer with kernel size 3, stride 2
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)

        # First Convolutional layer with 96 input channels, 256 output channels, kernel size 5, padding 2
        self.conv2 = nn.Conv2d(96,256,kernel_size=5,padding=2)

        # Second max Pooling layer with kernel size 3, stride 2
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)


        # Stage 2 ->  Pass three Convolutional layers then follow a Pooling layer

        # Third Convolutional layer with 256 input channels, 384 output channels, kernel size 3, padding 1
        self.conv3 = nn.Conv2d(256,384,kernel_size=3,padding=1)

        # Fourth Convolutional layer with 384 input channels, 384 output channels, kernel size 3, padding 1
        self.conv4 = nn.Conv2d(384,384,kernel_size=3,padding=1)

        # Fifth Convolutional layer with 384 input channels, 256 output channels, kernel size 3, padding 1
        self.conv5 = nn.Conv2d(384,256,kernel_size=3,padding=1)

        # Third max Pooling layer with kernel size 3, stride 2
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2)


        # Stage 3 -> Pass three Full Connected layers

        # First Fully connected layer with 4096 output features
        self.fc1 = nn.Linear(6*6*256,4096)

        # Second Fully connected layer with 4096 output features
        self.fc2 = nn.Linear(4096,4096)

        # Output layer with 1000 classes
        self.output = nn.Linear(4096,1000)

    def forward(self,x):

        # Forward pass through first convolutional layer with activation function
        x = F.relu(self.conv1(x))

        # Max pooling
        x = self.pool1(x)

        # Forward pass through second convolutional layer with activation function
        x = F.relu(self.conv2(x))

        # Max pooling
        x = self.pool2(x)

        # Forward pass through third convolutional layer with activation function
        x = F.relu(self.conv3(x))

        # Forward pass through fourth convolutional layer with activation function
        x = F.relu(self.conv4(x))

        # Forward pass through fifth convolutional layer with activation function
        x = F.relu(self.conv5(x))

        # Max pooling
        x = self.pool3(x)

        # Flatten the tensor to match the input shape required by the fully connected layer
        x = x.view(-1,6*6*256)
        
        # Forward pass through first fully connected layer with dropout and activation function
        x = F.relu(F.dropout(self.fc1(x),0.5))

        # Forward pass through second fully connected layer with dropout and activation
        x = F.relu(F.dropout(self.fc2(x),0.5))

        # Output layer with softmax activation function for classification
        output = F.softmax(self.output(x),dim=1)


        return output

# Instantiate the model
net = Model()

# Forward pass
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(dev)
data = data.to(dev)

# Get model output
output = net(data)
print(output)







