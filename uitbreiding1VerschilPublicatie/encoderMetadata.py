import os

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn

import torch.nn.functional as F
from transformers import BertTokenizer
from uitbreiding1VerschilPublicatie.OneHotEncoder import oneHotEncoder
from torch.utils.data import DataLoader

class encoderMetadata(nn.Module):
    # Create neural network
    def __init__(self,number_filters, kernelSize,oneHotEncoder):
        super(encoderMetadata, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.number_filters = number_filters
        self.kernelSize = kernelSize

        # Conv Network
        self.conv = nn.Conv1d(in_channels=oneHotEncoder.getLength(),
                      out_channels=self.number_filters,
                      kernel_size=self.kernelSize)#.to(self.device)



    def forward(self, metadata):
        # apply CNN and ReLu
        metadata_shaped = metadata.permute(0,2,1)
        metadata_conv = F.relu(self.conv(metadata_shaped))
        #max pooling
        metadata_pool = F.max_pool1d(metadata_conv, kernel_size=metadata_conv.shape[2])
        # Concatenate metadata_pool
        metadata_fc = metadata_pool.squeeze(dim=2)
        return metadata_fc

    '''
    Function for saving the neural network
    '''
    def saving_NeuralNetwork(self, path):
        torch.save(self.state_dict(), path)

    '''
    Function for loading the configurations from a file
    It first reads the configurations of a file
    Then it initialises a neural network with the parameters of the file
    Then it sets the neural network on the state of the loaded neural network
    '''
    def loading_NeuralNetwork(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

'''
Function for loading the configurations from a file
It first reads the configurations of a file
Then it initialises a neural network with the parameters of the file
Then it sets the neural network on the state of the loaded neural network
'''
def loading_NeuralNetwork(path, pathToConfigurations):
    configurations  =readConfigurations(pathToConfigurations)
    model = encoderMetadata(configurations['num_filters'][0],configurations['kernelSize'][0])
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

'''
Function for reading the configurations from a file
If the data is about the neural network, it converts it to an integer
In the other case it is written as a float
The data is saved in a dictionary
'''
def readConfigurations(pathToConfigurations):
    file = open(pathToConfigurations,'r')
    dict = {}
    Lines = file.readlines()

    for line in Lines:
        dict[line.split(':')[0]] = [int(num) for num in line.split(':')[1].split(',')]
    return dict
