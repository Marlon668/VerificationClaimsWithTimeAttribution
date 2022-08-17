import os
import re

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn

import torch.nn.functional as F
from transformers import BertTokenizer

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
        ).to(self.device)


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


