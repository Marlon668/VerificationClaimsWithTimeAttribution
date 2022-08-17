import collections
import os
import re
import numpy as np

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn

import torch.nn.functional as F
from transformers import BertTokenizer

from datasetOld import dump_load, dump_write, NUS
from torch.utils.data import DataLoader

class labelEmbeddingLayer(nn.Module):
    # Create neural network
    def __init__(self,embedding_dim, labelDomains):
        super(labelEmbeddingLayer, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.labelSequence = labelDomains
        self.word_embeds = nn.Embedding(115, self.embedding_dim)#.to(self.device)
        nn.init.xavier_uniform_(self.word_embeds.weight)


    def forward(self, input,domain):
        tokens = torch.tensor(range(115)).to(self.device)
        embeddings = self.word_embeds(tokens).to(self.device)
        label_distribution = torch.matmul(input,torch.transpose(embeddings,0,1)).to(self.device)
        return label_distribution
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
Function for loading the configurations from a file
It first reads the configurations of a file
Then it initialises a neural network with the parameters of the file
Then it sets the neural network on the state of the loaded neural network
'''
def loading_NeuralNetwork(path, pathToConfigurations):
    configurations  =readConfigurations(pathToConfigurations)
    model = labelEmbeddingLayer(configurations['embedding_dim'][0],configurations['hidden_dim'][0])
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def readLabels(path,pathToWrite):
    file = open(path, 'r')
    domains = dict()
    allLabels = set()
    lines = file.readlines()
    for line in lines:
        elements = line.split("\t")
        domains[elements[0]] = elements[1].replace('\n','').split(',')
        for label in domains.get(elements[0]):
            allLabels.add(label)
    labelSequence = list(allLabels)
    domainsNew = dict()
    for domain in domains.keys():
        labelsDomain = []
        for label in domains.get(domain):
            labelsDomain.append(labelSequence.index(label))
        domainsNew[domain] = labelsDomain
    file = open(pathToWrite, 'w',encoding="utf-8")
    for label in labelSequence:
        file.write(label + "\n")
    file.close()
    return labelSequence,domainsNew

def readLabelsForEachDomain(path,path2):
    labelsDomain = dict()
    domains = set()
    domainLabels = dict()
    file = open(path, 'r',encoding='utf-8')
    for claim in file:
        elements = claim.split('\t')
        domain = elements[0].split('-')[0]
        label = elements[2]
        if domain in domains:
            labelsDomain[domain].append(label)
        else:
            domains.add(domain)
            labelsDomain[domain] = list()
            labelsDomain[domain].append(label)
    file.close()
    file = open(path2, 'r', encoding='utf-8')
    for line in file:
        elements = line.split("\t")
        domainLabels[elements[0]] = elements[1].replace('\n', '').split(',')

    file = open('labels/weights.tsv', 'w', encoding='utf-8')
    for domain in domainLabels.keys():
        string = domain
        weightsDomain = np.zeros(len(domainLabels[domain]))
        max = collections.Counter(labelsDomain[domain]).most_common(1)[0][1]
        for i in range(len(domainLabels[domain])):
            print(domainLabels[domain][i])
            weightsDomain[i] = max/labelsDomain[domain].count(domainLabels[domain][i])
        for weight in weightsDomain:
            string += '\t' + str(weight)
        file.write(string + '\n')

        print(collections.Counter(labelsDomain[domain]))
    file.close()

    '''
    for line in file:
        elements = line.split("\t")
        domainLabels[elements[0]] = elements[1].replace('\n', '').split(',')
    for domain in domainLabels.keys():
        for label in domainLabels[domain]:
            if label not in labelsDomain[domain]:
                print('domain : ' + domain + ' - label : ' + label)
    '''
    return labelsDomain
'''
Function for reading the configurations from a file
If the data is about the neural network, it converts it to an integer
In the other case it is written as a float
The data is saved in a dictionary
'''
def readConfigurations(pathToConfigurations):
    file = open(pathToConfigurations,'r',encoding='utf-8')
    dict = {}
    Lines = file.readlines()

    for line in Lines:
        dict[line.split(':')[0]] = [int(num) for num in line.split(':')[1].split(',')]
    return dict


'''
This is the train-loop for training the neural network
#input: dataLoader: the dataloader of the test-set
#       model: the pre-trained network to evaluate
#       valueC: the value C used in the loss function
#       chosenLoss: the used loss function, this could be:
#                     1: crossModalTripletLossRandom (see Loss_function.py)
#                     2: crossModalTripletLossHardNegativeSampling (see Loss_function.py)
#                     3: softWeightedTripletLoss (see Loss_function.py)
#                     4: softMarginTripletLoss (see Loss_function.py)
#                     5: softMarginTripletLossRandom (see Loss_function.py)
#                     6: softMarginTripletLossOtherHardNegativeSampling (see Loss_function.py)
#                     7: softMarginTripletLossRandomByShift (see Loss_function.py)
#                     8: softMarginTripletLossByTakingAll (see Loss_function.py)
#                     9: softMarginTripletLossHardSampling2 (see Loss_function.py)
#                    10: softMarginTripletLossRandom2 (see Loss_function.py)
#                    11: softMarginTripletLossOtherHardNegativeSampling2 (see Loss_function.py)
#                    12: softMarginTripletLossRandomByShift2 (see Loss_function.py)
#                    13: softMarginTripletLossByTakingAll2 (see Loss_function.py)
'''
def train_loop(dataloader, model):
    size = len(dataloader.dataset)
    totalLoss = 0
    # Bert transformer for embedding the word captions
    #transformer = SentenceTransformer('paraphrase-distilroberta-basisModel-v1')
    for c in dataloader:
        # Compute prediction and loss
        # outcome of feedforwarding feature vector to image neural network
        # set gradient to true
        for snippet in c[2]:
            pred1= model(snippet[1])

    #print('Average loss : ' ,  totalLoss/size)

if __name__ == "__main__":
    #readLabels('labels/labels.tsv','labels/labelSequence')
    print(readLabelsForEachDomain("train/train.tsv","labels/labels.tsv"))
