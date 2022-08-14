import os

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn

import torch.nn.functional as F
from transformers import AutoTokenizer

from datasetOld import dump_load, dump_write, NUS
from torch.utils.data import DataLoader

class encoderClaim(nn.Module):
    # Create neural network
    def __init__(self,embedding_dim, hidden_dim,number_layers=2,drop_out=0.1):
        super(encoderClaim, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeds = nn.Embedding(self.tokenizer.vocab_size, embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.word_embeds.weight)
        self.forwardLSTM = nn.ModuleList().to(self.device)
        self.backwardLSTM = nn.ModuleList().to(self.device)
        for i in range(number_layers):
            input_size = embedding_dim if i == 0 else hidden_dim
            self.forwardLSTM.append(nn.LSTM(input_size, hidden_dim, num_layers=1)).to(self.device)
            self.backwardLSTM.append(nn.LSTM(input_size, hidden_dim, num_layers=1)).to(self.device)

        self.claimDate = nn.Embedding(2,2*hidden_dim).to(self.device)
        self.evidenceDate = nn.Embedding(65, 2 * hidden_dim).to(self.device)
        self.dropout = nn.Sequential(
            torch.nn.Dropout(p=drop_out),
            torch.nn.ReLU(),
        ).to(self.device)
        self.number_layers = number_layers
        self.skip= nn.Sequential(
            nn.Identity(),
            torch.nn.Dropout(p=drop_out),
            torch.nn.ReLU(),
        ).to(self.device)


    def forward(self, claim,date,isClaim = True):
        encoded_input = self.tokenizer(claim, padding=True, truncation=False, return_tensors='pt').to(self.device)
        #tokens = [[i for i in self.tokenizer(claim)['input_ids'] if i not in {0,2,4}]]
        #tokens = torch.tensor(tokens).to(self.device)
        inputForward = self.dropout(self.word_embeds(encoded_input['input_ids']).to(self.device))
        inputBackward = torch.flip(inputForward,[1]).to(self.device)
        outputForwards = torch.tensor([]).to(self.device)
        outputBackWards = torch.tensor([]).to(self.device)
        alpha = 0.9
        for i in range(self.number_layers):
            if i != 0:
                inputForward = self.dropout(inputForward)
                inputBackward = self.dropout(inputBackward)
                #skip connections
                for j in range(i):
                    inputForward = inputForward + self.skip(outputForwards[j])
                    inputBackward = inputBackward + self.skip(outputBackWards[j])

            outputForward, hiddenForward = self.forwardLSTM[i](inputForward)
            outputBackWard, hiddenBackward = self.backwardLSTM[i](inputBackward)
            outputForwards = torch.cat((outputForwards,outputForward))
            outputBackWards = torch.cat((outputBackWards,outputBackWard))
            inputForward = outputForward
            inputBackward = outputBackWard
        if isClaim:
            return alpha * torch.cat((self.dropout(outputForward[0][-1]),self.dropout(outputBackWard[0][-1]))) + (1-alpha)*self.claimDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)
        else:
            return alpha * torch.cat((self.dropout(outputForward[0][-1]), self.dropout(outputBackWard[0][-1]))) + (1-alpha)* self.evidenceDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)


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
