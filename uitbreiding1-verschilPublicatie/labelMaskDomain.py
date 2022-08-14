import os
import re

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn

import torch.nn.functional as F
from transformers import BertTokenizer

from dataset import dump_load, dump_write, NUS
from torch.utils.data import DataLoader

class labelMaskDomain(nn.Module):
    # Create neural network
    def __init__(self,embedding_dim, labelDomains,domain,inut_size,hiddenDimension=100):
        super(labelMaskDomain, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.tokens = torch.tensor(labelDomains[domain]).to(self.device)
        self.input_size = inut_size
        self.label = nn.Sequential(
            nn.Linear(inut_size, hiddenDimension),
            torch.nn.LeakyReLU(),
            nn.Linear(hiddenDimension,inut_size)
        )#.to(self.device)


    def forward(self, input):
        label_distribution_domain = torch.take(input,self.tokens)
        return self.label(label_distribution_domain)


    '''
    Function for saving the neural network
    '''
    def saving_NeuralNetwork(self, path):
        torch.save(self.state_dict(), path)
        return self

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
