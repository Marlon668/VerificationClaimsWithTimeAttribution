import os

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn

import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import DataLoader

class instanceEncoder(nn.Module):
    # Create neural network
    def __init__(self):
        super(instanceEncoder, self).__init__()




    def forward(self, claim_encoding,evidence_encoding,metadata_encoding):
        return torch.cat((claim_encoding,evidence_encoding,claim_encoding-evidence_encoding,torch.dot(claim_encoding,evidence_encoding).unsqueeze(0),metadata_encoding))


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
    def loading_NeuralNetwork(self,path):
        self.load_state_dict(torch.load(path))
        self.eval()

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
