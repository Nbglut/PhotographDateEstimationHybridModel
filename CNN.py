"""
define moduals of model
"""
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

#Conv2dBlock for Encoder
class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_rate):
        super().__init__()
        self.conv = nn.Sequential(
             nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, stride=stride, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Conv1d(in_channels=32, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),

                      )
        self.conv.apply(add_weights)        



    def forward(self, x):  #default  
       # print("Input shape:", x.shape)  # Before convolution
        x = x.permute(0, 2, 1)
        result = self.conv(x)        
       # print("Output shape:", result.shape)  # After convolution
        result = result.permute(0, 2, 1)

        return result
    

        
def add_weights(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
        
                
    



class CNNModel(nn.Module):
    """docstring for ClassName"""
    
    def __init__(self, args):
        super(CNNModel, self).__init__()
        ##-----------------------------------------------------------
        ##
        ##-----------------------------------------------------------

        ## define CNN layers below
        self.conv = nn.Sequential( ## Layer1
                    nn.Conv1d(in_channels=32,out_channels=64, kernel_size=4, stride=2),
                    #                nn.Conv2d(in_channels=3,out_channels=200, kernel_size=2, stride=2),
                    nn.BatchNorm1d(64),

                    nn.ReLU(),
                    nn.Dropout(0.2),
                    #Layer 2
                    nn.Conv1d(in_channels=64,out_channels=128, kernel_size=4, stride=2),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(2,2),
                    nn.Dropout(0.2),
                    #Layer 3
                    nn.Conv1d(in_channels=128,out_channels=256, kernel_size=3, stride=2),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Flatten(),
                    #fc1
                    nn.Linear(256, 1225),
                    nn.BatchNorm1d(1225),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    #fc2
                    nn.Linear(1225, 13)
    
                      )
        self.conv.apply(add_weights)        




    '''feed features to the model'''
    def forward(self, x):  #default
        
        ##---------------------------------------------------------
        ## write code to feed input features to the CNN models defined above
        ##---------------------------------------------------------
        result = self.conv(x)

        ## write flatten tensor code below
    #x =  torch.flatten(x_out)
        

        ## ---------------------------------------------------
        ## write fully connected layer (Linear layer) below
        ## ---------------------------------------------------
        #x = self.fc(x)  # predict y
        #result=self.fc1(x)
        
        return result
    

        
def add_weights(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
        
                
    
        
