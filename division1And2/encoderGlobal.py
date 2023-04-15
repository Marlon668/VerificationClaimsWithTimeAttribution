import os

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn

import torch.nn.functional as F
from transformers import AutoTokenizer

from torch.utils.data import DataLoader

class encoder(nn.Module):
    # Create neural network
    def __init__(self,embedding_dim, hidden_dim,alpha,beta,number_layers=2,drop_out=0.1):
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
        self.hidden_dim = hidden_dim
        self.positionEmbeddings = nn.Embedding(200, embedding_dim).to(self.device)
        self.predicateEmbeddings = nn.Embedding(2, embedding_dim).to(self.device)
        self.verschil = nn.Embedding(86, 2 * hidden_dim).to(self.device)
        self.claimDate = nn.Embedding(2, 2 * hidden_dim).to(self.device)
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
        self.beta = float(beta)


    def forward(self, claim,date,positions,verbs,times,verschillenIndices,verschillenValues,sizePretext=0,isClaim = True):
        encoded_input = self.tokenizer(claim, padding=True, truncation=False, return_tensors='pt').to(self.device)
        inputForward = self.word_embeds(encoded_input['input_ids']).to(self.device)
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
        output = torch.cat((self.dropout(outputForward[0][-1]), self.dropout(outputBackWard[0][-1])))
        tijdAbsolute = torch.zeros(2 * self.hidden_dim).squeeze(0).to(self.device)
        number = 0
        if verschillenIndices[0]!="":
            for i in range(len(verschillenIndices)):
                if verschillenIndices[i] >= sizePretext and verschillenValues[i].find('Duur')==-1 and verschillenValues[i].find('Refs')==-1:
                    if verschillenValues[i].isdigit():
                        tijdAbsolute  += self.verschil(torch.tensor([int(verschillenValues[i])]).to(self.device)).squeeze(0).to(self.device)
                        number += 1
                if i+1 >= len(verschillenValues):
                    break
        if isClaim:
            verschilDatum = self.claimDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)
            if number > 0:
                return self.alpha * output + self.beta * verschilDatum + (1 - self.alpha - self.beta) * tijdAbsolute / number
            else:
                return self.alpha * output + (1 - self.alpha) * verschilDatum
        else:
            verschilDatum = self.evidenceDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)
            if number > 0:
                return self.alpha * output + self.beta * verschilDatum + (1 - self.alpha - self.beta) * tijdAbsolute / number
            else:
                return self.alpha * output + (1 - self.alpha) * verschilDatum

    def forwardAttribution(self, claim,date,positions,verbs,times,verschillenIndices,verschillenValues,sizePretext = 0, isClaim = True):
        encoded_input = self.tokenizer(claim, padding=True, truncation=False, return_tensors='pt').to(self.device)
        inputForward = self.word_embeds(encoded_input['input_ids']).to(self.device)
        inputForward = torch.nn.functional.normalize(inputForward, p=2.0)
        inputForward = self.dropout(inputForward)
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
            outputBackWard, hiddenBackward = self.backwardLSTM[i](inputBackward)
            outputForwards = torch.cat((outputForwards, outputForward))
            outputBackWards = torch.cat((outputBackWards, outputBackWard))
            inputForward = outputForward
            inputBackward = outputBackWard
        output = torch.cat((self.dropout(outputForward[0][-1]), self.dropout(outputBackWard[0][-1])))

        tijdAbsolute = torch.zeros(2 * self.hidden_dim).squeeze(0).to(self.device)
        times = []
        number = 0
        if verschillenIndices[0]!="":
            for i in range(len(verschillenIndices)):
                if verschillenIndices[i] >= sizePretext and verschillenValues[i].find('Duur')==-1 and verschillenValues[i].find('Refs')==-1:
                    if verschillenValues[i].isdigit():
                        time = self.verschil(torch.tensor([int(verschillenValues[i])]).to(self.device)).squeeze(0).to(self.device)
                        tijdAbsolute  += time
                        number += 1
                        times.append(time)
                if i+1 >= len(verschillenValues):
                    break

        if isClaim:
            verschilDatum = self.claimDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)
            if number > 0:
                return self.alpha * output + self.beta * verschilDatum + (1 - self.alpha - self.beta) * tijdAbsolute / number,\
                        output,times,verschilDatum
            else:
                return self.alpha * output + (1 - self.alpha) * verschilDatum,output,times,verschilDatum
        else:
            verschilDatum = self.evidenceDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)
            if number > 0:
                return self.alpha * output + self.beta * verschilDatum + (1 - self.alpha - self.beta) * tijdAbsolute / number,\
                        output,times,verschilDatum
            else:
                return self.alpha * output + (1 - self.alpha) * verschilDatum,output,times,verschilDatum

    def addTime(self,claim,encoding,date,positions,verbs,times,verschillenIndices,verschillenValues,sizePretext=0, isClaim = True):
        tijdAbsolute = torch.zeros(2 * self.hidden_dim).squeeze(0).to(self.device)
        number = 0
        if verschillenIndices[0] != "":
            for i in range(len(verschillenIndices)):
                if verschillenIndices[i] >= sizePretext and verschillenValues[i].find('Duur') == -1 and verschillenValues[i].find('Refs') == -1:
                    if verschillenValues[i].isdigit():
                        tijdAbsolute += self.verschil(
                            torch.tensor([int(verschillenValues[i])]).to(self.device)).squeeze(0).to(self.device)
                        number += 1
                if i + 1 >= len(verschillenValues):
                    break
        if isClaim:
            verschilDatum = self.claimDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)
            if number > 0:
                return self.alpha * encoding + self.beta * verschilDatum + (1 - self.alpha - self.beta) * tijdAbsolute / number
            else:
                return self.alpha * encoding + (1 - self.alpha) * verschilDatum
        else:
            verschilDatum = self.evidenceDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)
            if number > 0:
                return self.alpha * encoding + self.beta * verschilDatum + (1 - self.alpha - self.beta) * tijdAbsolute / number
            else:
                return self.alpha * encoding + (1 - self.alpha) * verschilDatum

    def getTimeEncodings(self,claim,encoding,date,positions,verbs,times,verschillenIndices,verschillenValues,sizePretext=0,isClaim = True):
        times = []
        if verschillenIndices[0] != "":
            for i in range(len(verschillenIndices)):
                if verschillenIndices[i] >= sizePretext and verschillenValues[i].find('Duur') == -1 and verschillenValues[i].find('Refs') == -1:
                    if verschillenValues[i].isdigit():
                        times.append(self.verschil(
                            torch.tensor([int(verschillenValues[i])]).to(self.device)).squeeze(0).to(self.device))
                if i + 1 >= len(verschillenValues):
                    break
        if isClaim:
            return times,self.claimDate(torch.tensor([date]).to(self.device))
        else:
            return times,self.evidenceDate(torch.tensor([date]).to(self.device))



    def getEncodingWithoutTime(self,claim):
        encoded_input = self.tokenizer(claim, padding=True, truncation=False, return_tensors='pt').to(self.device)
        inputForward = self.dropout(self.word_embeds(encoded_input['input_ids']).to(self.device))
        inputBackward = torch.flip(inputForward, [1]).to(self.device)
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
    Function for loading the neural network
    '''
    def loading_NeuralNetwork(self,path):
        self.load_state_dict(torch.load(path))
        self.eval()