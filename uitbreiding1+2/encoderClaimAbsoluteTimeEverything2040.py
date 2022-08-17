import os

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn

import torch.nn.functional as F
from transformers import AutoTokenizer

from datasetOld import dump_load, dump_write, NUS
from torch.utils.data import DataLoader

class encoderAbsolute(nn.Module):
    # Create neural network
    def __init__(self,embedding_dim, hidden_dim,number_layers=2,drop_out=0.1):
        super(encoderAbsolute, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-basisModel-v1')
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


    def forward(self, claim,date,positions,verbs,times,verschillenIndices,verschillenValues,isClaim = True):
        encoded_input = self.tokenizer(claim, padding=True, truncation=False, return_tensors='pt').to(self.device)
        inputForward = self.word_embeds(encoded_input['input_ids']).to(self.device)
        inputForward = torch.nn.functional.normalize(inputForward,p=2.0)
        inputForward = self.dropout(inputForward)
        inputBackward = torch.flip(inputForward,[1]).to(self.device)
        outputForwards = torch.tensor([]).to(self.device)
        outputBackWards = torch.tensor([]).to(self.device)
        alpha = 0.2
        beta = 0.4
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
                if verschillenValues[i].find('Duur')==-1 and verschillenValues[i].find('Refs')==-1:
                    if verschillenValues[i].isdigit():
                        tijdAbsolute  += self.verschil(torch.tensor([int(verschillenValues[i])]).to(self.device)).squeeze(0).to(self.device)
                        number += 1
                if i+1 >= len(verschillenValues):
                    break
        if isClaim:
            verschilDatum = self.claimDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)
            if number > 0:
                return alpha * output + beta * verschilDatum + (1 - alpha - beta) * tijdAbsolute / number
            else:
                return alpha * output + (1 - alpha) * verschilDatum
        else:
            verschilDatum = self.evidenceDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)
            if number > 0:
                return alpha * output + beta * verschilDatum + (1 - alpha - beta) * tijdAbsolute / number
            else:
                return alpha * output + (1 - alpha) * verschilDatum

    def forwardAttribution(self, claim,date,positions,verbs,times,verschillenIndices,verschillenValues,isClaim = True):
        encoded_input = self.tokenizer(claim, padding=True, truncation=False, return_tensors='pt').to(self.device)
        inputForward = self.word_embeds(encoded_input['input_ids']).to(self.device)
        inputForward = torch.nn.functional.normalize(inputForward, p=2.0)
        inputForward = self.dropout(inputForward)
        inputBackward = torch.flip(inputForward, [1]).to(self.device)
        outputForwards = torch.tensor([]).to(self.device)
        outputBackWards = torch.tensor([]).to(self.device)
        alpha = 0.2
        beta = 0.4
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

        #outputForward = outputForward.normalise

        #outputForward = torch.nn.functional.normalize(outputForward,p=1.0)
        #outputBackWard = torch.nn.functional.normalize(outputBackWard, p=1.0)
        encodingClaim = torch.cat((self.dropout(outputForward[0][-1]),self.dropout(outputBackWard[0][-1])))
        tijdAbsolute = torch.zeros(2 * self.hidden_dim).squeeze(0).to(self.device)
        times = []
        number = 0
        if verschillenIndices[0]!="":
            for i in range(len(verschillenIndices)):
                if verschillenValues[i].find('Duur')==-1 and verschillenValues[i].find('Refs')==-1:
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
                return alpha * output + beta * verschilDatum + (1 - alpha - beta) * tijdAbsolute / number,\
                        output,times,verschilDatum
            else:
                return alpha * output + (1 - alpha) * verschilDatum,output,times,verschilDatum
        else:
            verschilDatum = self.evidenceDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)
            if number > 0:
                return alpha * output + beta * verschilDatum + (1 - alpha - beta) * tijdAbsolute / number,\
                        output,times,verschilDatum
            else:
                return alpha * output + (1 - alpha) * verschilDatum,output,times,verschilDatum

    def addTime(self,claim,encoding,date,positions,verbs,times,verschillenIndices,verschillenValues,isClaim = True):
        alpha = 0.2
        beta = 0.4
        tijdAbsolute = torch.zeros(2 * self.hidden_dim).squeeze(0).to(self.device)
        number = 0
        if verschillenIndices[0] != "":
            for i in range(len(verschillenIndices)):
                if verschillenValues[i].find('Duur') == -1 and verschillenValues[i].find('Refs') == -1:
                    if verschillenValues[i].isdigit():
                        tijdAbsolute += self.verschil(
                            torch.tensor([int(verschillenValues[i])]).to(self.device)).squeeze(0).to(self.device)
                        number += 1
                if i + 1 >= len(verschillenValues):
                    break
        if isClaim:
            verschilDatum = self.claimDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)
            if number > 0:
                return alpha * encoding + beta * verschilDatum + (1 - alpha - beta) * tijdAbsolute / number
            else:
                return alpha * encoding + (1 - alpha) * verschilDatum
        else:
            verschilDatum = self.evidenceDate(torch.tensor([date]).to(self.device)).squeeze(0).to(self.device)
            if number > 0:
                return alpha * encoding + beta * verschilDatum + (1 - alpha - beta) * tijdAbsolute / number
            else:
                return alpha * encoding + (1 - alpha) * verschilDatum

    def getTimeEncodings(self,claim,encoding,date,positions,verbs,times,verschillenIndices,verschillenValues,isClaim = True):
        times = []
        if verschillenIndices[0] != "":
            for i in range(len(verschillenIndices)):
                if verschillenValues[i].find('Duur') == -1 and verschillenValues[i].find('Refs') == -1:
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
    Function for loading the configurations from a file
    It first reads the configurations of a file
    Then it initialises a neural network with the parameters of the file
    Then it sets the neural network on the state of the loaded neural network
    '''
    def loading_NeuralNetwork(self,path):
        self.load_state_dict(torch.load(path))
        self.eval()


if __name__ == "__main__":

    # loading the configurations of the neural network
    model = encoderTokens(100,10)
    # Loading in the train-set

    train_set = dump_load('basisModel/trainLoader')

    # dataloader for the train-set
    train_loader = DataLoader(train_set,
                           batch_size=int(1),
                           shuffle=True)


    # dataloader for the test-set
    #number of epochs
    epochs = 1
    # This is the bestAcc we have till now
    bestAcc = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # Train loop
        train_loop(train_loader, model)