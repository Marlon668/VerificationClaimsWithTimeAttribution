import os

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn

import torch.nn.functional as F
from transformers import AutoTokenizer

from torch.utils.data import DataLoader

class encoder(nn.Module):
    # Create neural network
    def __init__(self,embedding_dim, hidden_dim,alpha,number_layers=2,drop_out=0.1):
        super(encoder, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
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
        self.alpha = float(alpha)


    def forward(self, claim,date,isClaim = True):
        encoded_input = self.tokenizer(claim, padding=True, truncation=False, return_tensors='pt').to(self.device)
        inputForward = self.dropout(self.word_embeds(encoded_input['input_ids']).to(self.device))
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
        if isClaim:
            return self.alpha * torch.cat((self.dropout(outputForward[0][-1]),self.dropout(outputBackWard[0][-1]))) + (1-self.alpha)*self.claimDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)
        else:
            return self.alpha * torch.cat((self.dropout(outputForward[0][-1]), self.dropout(outputBackWard[0][-1]))) + (1-self.alpha)* self.evidenceDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)

    def forwardAttribution(self, claim, date, isClaim=True):
        encoded_input = self.tokenizer(claim, padding=True, truncation=False, return_tensors='pt').to(self.device)
        # tokens = [[i for i in self.tokenizer(claim)['input_ids'] if i not in {0,2,4}]]
        # tokens = torch.tensor(tokens).to(self.device)
        inputForward = self.dropout(self.word_embeds(encoded_input['input_ids']).to(self.device))
        # inputForward = torch.nn.functional.normalize(inputForward, p=1.0)
        inputBackward = torch.flip(inputForward, [1]).to(self.device)
        outputForwards = torch.tensor([]).to(self.device)
        outputBackWards = torch.tensor([]).to(self.device)
        for i in range(self.number_layers):
            if i != 0:
                inputForward = self.dropout(inputForward)
                inputBackward = self.dropout(inputBackward)
                # skip connections
                for j in range(i):
                    inputForward = inputForward + self.skip(outputForwards[j])
                    inputBackward = inputBackward + self.skip(outputBackWards[j])

            outputForward, hiddenForward = self.forwardLSTM[i](inputForward)
            outputBackWard, hiddenBackward = self.forwardLSTM[i](inputBackward)
            outputForwards = torch.cat((outputForwards, outputForward))
            outputBackWards = torch.cat((outputBackWards, outputBackWard))
            inputForward = outputForward
            inputBackward = outputBackWard

        # outputForward = outputForward.normalise
        if isClaim:
            # outputForward = torch.nn.functional.normalize(outputForward,p=1.0)
            # outputBackWard = torch.nn.functional.normalize(outputBackWard, p=1.0)
            encodingClaim = torch.cat((self.dropout(outputForward[0][-1]), self.dropout(outputBackWard[0][-1])))
            time = self.claimDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)
            return self.alpha * encodingClaim + (1 - self.alpha) * time, \
                   encodingClaim, time
        else:
            # outputForward = torch.nn.functional.normalize(outputForward, p=1.0)
            # outputBackWard = torch.nn.functional.normalize(outputBackWard, p=1.0)
            encodingEvidence = torch.cat((self.dropout(outputForward[0][-1]), self.dropout(outputBackWard[0][-1])))
            time = self.evidenceDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)
            return self.alpha * encodingEvidence + (1 - self.alpha) * time, \
                   encodingEvidence, time

    def addTime(self, encoding, date, isClaim=True):
        if isClaim:
            return self.alpha * encoding + (1 - self.alpha) * self.claimDate(torch.tensor([date]).to(self.device)).squeeze(0).to(
                self.device)
        else:
            return self.alpha * encoding + (1 - self.alpha) * self.evidenceDate(torch.tensor([date]).to(self.device)).squeeze(
                0).to(self.device)

    def getTimeEncoding(self, date, isClaim=True):
        if isClaim:
            return self.claimDate(torch.tensor([date]).to(self.device))
        else:
            return self.evidenceDate(torch.tensor([date]).to(self.device))

    def getEncodingWithoutTime(self, claim):
        encoded_input = self.tokenizer(claim, padding=True, truncation=False, return_tensors='pt').to(self.device)
        inputForward = self.dropout(self.word_embeds(encoded_input['input_ids']).to(self.device))
        inputBackward = torch.flip(inputForward, [1]).to(self.device)
        outputForwards = torch.tensor([]).to(self.device)
        outputBackWards = torch.tensor([]).to(self.device)
        for i in range(self.number_layers):
            if i != 0:
                inputForward = self.dropout(inputForward)
                inputBackward = self.dropout(inputBackward)
                # skip connections
                for j in range(i):
                    inputForward = inputForward + self.skip(outputForwards[j])
                    inputBackward = inputBackward + self.skip(outputBackWards[j])

            outputForward, hiddenForward = self.forwardLSTM[i](inputForward)
            outputBackWard, hiddenBackward = self.forwardLSTM[i](inputBackward)
            outputForwards = torch.cat((outputForwards, outputForward))
            outputBackWards = torch.cat((outputBackWards, outputBackWard))
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
