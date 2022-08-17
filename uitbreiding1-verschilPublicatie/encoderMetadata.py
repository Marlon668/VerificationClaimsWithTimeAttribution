import os

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn

import torch.nn.functional as F
from transformers import BertTokenizer
from OneHotEncoder import oneHotEncoder
from datasetOld import dump_load, dump_write, NUS
from torch.utils.data import DataLoader

class encoderMetadata(nn.Module):
    # Create neural network
    def __init__(self,number_filters, kernelSize,oneHotEncoder):
        super(encoderMetadata, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.number_filters = number_filters
        self.kernelSize = kernelSize

        # Conv Network
        self.conv = nn.Conv1d(in_channels=oneHotEncoder.getLength(),
                      out_channels=self.number_filters,
                      kernel_size=self.kernelSize)#.to(self.device)



    def forward(self, metadata):
        # apply CNN and ReLu
        metadata_shaped = metadata.permute(0,2,1)
        metadata_conv = F.relu(self.conv(metadata_shaped))
        #max pooling
        metadata_pool = F.max_pool1d(metadata_conv, kernel_size=metadata_conv.shape[2])
        # Concatenate metadata_pool
        metadata_fc = metadata_pool.squeeze(dim=2)
        return metadata_fc

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
    def loading_NeuralNetwork(self, path):
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
    model = encoderMetadata(configurations['num_filters'][0],configurations['kernelSize'][0])
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

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
def train_loop(dataloader, model,oneHotEncoder):
    size = len(dataloader.dataset)
    totalLoss = 0
    # Bert transformer for embedding the word captions
    #transformer = SentenceTransformer('paraphrase-distilroberta-basisModel-v1')
    for c in dataloader:
        # Compute prediction and loss
        # outcome of feedforwarding feature vector to image neural network
        # set gradient to true
        input = []
        input.append(oneHotEncoder.encode(c[3]))
        input.append(oneHotEncoder.encode(c[4]))
        input.append(oneHotEncoder.encode(c[5]))
        input.append(oneHotEncoder.encode(c[6]))
        input = torch.stack(input,1)
        print(input)
        pred1= model(input)
        print(pred1.size())

    #print('Average loss : ' ,  totalLoss/size)

if __name__ == "__main__":

    oneHotEncoder = oneHotEncoder('basisModel/Metadata_sequence/all.txt')
    # loading the configurations of the neural network
    model = encoderMetadata(4,4,oneHotEncoder)
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
        train_loop(train_loader, model,oneHotEncoder)