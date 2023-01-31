import os

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn

import torch.nn.functional as F
from transformers import BertTokenizer
from division1DifferencePublication.OneHotEncoder import oneHotEncoder
from torch.utils.data import DataLoader

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)

class evidenceRanker(nn.Module):
    # Create neural network
    def __init__(self,sizeInput,hiddenDimension):
        super(evidenceRanker, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.rank = nn.Sequential(
            nn.Linear(sizeInput, hiddenDimension),
            torch.nn.ReLU(),
            nn.Linear(hiddenDimension, 50),
            torch.nn.ReLU(),
            nn.Linear(50, 1),
        )#.to(self.device)
        #initialising weights Xavier uniform
        self.rank.apply(init_weights)



    def forward(self, instance_encoding):
        rank = self.rank(instance_encoding)
        return rank

    '''
    Function for saving the neural network
    '''
    def saving_NeuralNetwork(self,path):
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
