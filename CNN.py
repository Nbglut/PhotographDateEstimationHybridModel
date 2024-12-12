"""
define moduals of model
"""
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

class ConvBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, kernel_size, stride, dropout_rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, mlp_dim, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm1d(mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(mlp_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x):
        return x + self.conv(x)



    def forward(self, x):  #default  
       # print("Input shape:", x.shape)  # Before convolution
        x = x.permute(0, 2, 1)
        result = self.conv(x)        
       # print("Output shape:", result.shape)  # After convolution
        result = result.permute(0, 2, 1)

        return result
    
class ConvEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = ConvBlock(hidden_dim, mlp_dim,3,1, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class ConvEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = ConvEncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )

        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))



        
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
        
                
    
        
