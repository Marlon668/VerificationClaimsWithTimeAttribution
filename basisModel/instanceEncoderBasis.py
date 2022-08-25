import os

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn

import torch.nn.functional as F
from transformers import BertTokenizer

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
    '''
    def loading_NeuralNetwork(self,path):
        self.load_state_dict(torch.load(path))
        self.eval()
