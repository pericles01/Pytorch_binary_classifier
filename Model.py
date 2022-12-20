import torch.nn as nn
import torch

class Binary_Classifier(nn.Module):
    # 3 

    def __init__(self, input, num_classes):
        super(Binary_Classifier, self).__init__()
        
       

        self.block1 = self.conv_block(c_in=input, c_out=256, dropout=0.25, kernel_size=5)
        self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.25, kernel_size=3)
        self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.25, kernel_size=3)
        #self.block4 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=56)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.lastlayer = nn.Linear(64*3*3, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.lastlayer(x)
        x = nn.Sigmoid(x)

        return x

    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block