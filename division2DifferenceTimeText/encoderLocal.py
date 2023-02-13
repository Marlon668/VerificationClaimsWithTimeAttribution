import os

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn

import torch.nn.functional as F
from transformers import AutoTokenizer

from torch.utils.data import DataLoader

class encoder(nn.Module):
    # Create neural network
    def __init__(self,embedding_dim, hidden_dim,alpha,withPretext=False,number_layers=2,drop_out=0.1):
        super(encoder, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeds = nn.Embedding(self.tokenizer.vocab_size, embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.word_embeds.weight)
        self.forwardLSTM = nn.ModuleList().to(self.device)
        self.backwardLSTM = nn.ModuleList().to(self.device)
        self.withPretext = withPretext
        for i in range(number_layers):
            input_size = embedding_dim if i == 0 else hidden_dim
            self.forwardLSTM.append(nn.LSTM(input_size, hidden_dim, num_layers=1)).to(self.device)
            self.backwardLSTM.append(nn.LSTM(input_size, hidden_dim, num_layers=1)).to(self.device)

        self.positionEmbeddings = nn.Embedding(200,embedding_dim).to(self.device)
        self.predicateEmbeddings = nn.Embedding(2,embedding_dim).to(self.device)
        self.verschil = nn.Embedding(86,embedding_dim).to(self.device)
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
        self.batch = 0
        self.alpha = float(alpha)


    def forward(self, text,date,positions,verbs,times,verschillenIndices,verschillenValues,claimId, snippetNumber=0,train=False,isClaim = True):
        encoded_input = self.tokenizer(text, padding=True, truncation=False, return_tensors='pt').to(self.device)
        sizePrexext = 0
        if self.withPretext:
            if isClaim:
                pretextF = open("r",)

            else:
                pretextF = open("r", )
        inputForward = self.word_embeds(encoded_input['input_ids']).to(self.device)
        if positions[0]!="":
            for position in positions:
                position = position.split(',')
                inputForward[0][int(position[0])-sizePrexext] = self.alpha*inputForward[0][int(position[0])-sizePrexext] + (1-self.alpha)*self.positionEmbeddings(torch.tensor([int(position[1])+100]).to(self.device)).squeeze(0).to(self.device)
        if verbs[0]!="":
            for verb in verbs:
                inputForward[0][int(verb)-sizePrexext] = self.alpha*inputForward[0][int(verb)-sizePrexext] + (1-self.alpha)*self.predicateEmbeddings(torch.tensor([0]).to(self.device)).squeeze(0).to(self.device)
        if times[0]!="":
            for time in times:
                inputForward[0][int(time)-sizePrexext] = self.alpha*inputForward[0][int(time)-sizePrexext] + (1-self.alpha)*self.predicateEmbeddings(torch.tensor([1]).to(self.device)).squeeze(0).to(self.device)
        if verschillenIndices[0]!="":
            for i in range(len(verschillenIndices)):
                index = verschillenIndices[i]
                if verschillenValues[i].find('Duur')==-1 and verschillenValues[i].find('Refs')==-1:
                    if verschillenValues[i].isdigit():
                        inputForward[0][int(index)-sizePrexext] = self.alpha*inputForward[0][int(index)-sizePrexext] + (1-self.alpha)* self.verschil(torch.tensor([int(verschillenValues[i])]-sizePrexext).to(self.device)).squeeze(0).to(self.device)
                if i+1 >= len(verschillenValues):
                    break
        inputForward = torch.nn.functional.normalize(inputForward,p=2.0)
        inputForward = self.dropout(inputForward)
        inputBackward = torch.flip(inputForward,[1]).to(self.device)
        outputForwards = torch.tensor([]).to(self.device)
        outputBackWards = torch.tensor([]).to(self.device)
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
        return torch.cat((self.dropout(outputForward[0][-1]), self.dropout(outputBackWard[0][-1])))

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
