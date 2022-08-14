import os
import sys
import random

import numpy as np
from pytorchtools import EarlyStopping
import torch.nn.functional as F

import torch
from torch.utils.data import DataLoader

from labelMaskDomain import labelMaskDomain
from datasetIteratie2CombinerOld import dump_load, dump_write, NUS
from encoderClaim import encoderClaim
from encoderEvidence import encoderEvidence
from encoderMetadata import encoderMetadata
from evidence_ranker import evidenceRanker
from instanceEncoder import instanceEncoder
from labelEmbeddingLayer import labelEmbeddingLayer
from OneHotEncoder import oneHotEncoder
from torch import nn
from sklearn.metrics import f1_score
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer

from transformers import BertTokenizer, AutoModel, AutoTokenizer


class verifactionModel(nn.Module):
    # Create neural network
    def __init__(self,transformer,metadataEncoder,instanceEncoder,evidenceRanker,labelEmbedding,labelMaskDomain,labelDomains,domainWeights,domain):
        super(verifactionModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
        self.transformer = transformer
        self.metaDataEncoder = metadataEncoder
        self.instanceEncoder = instanceEncoder
        self.evidenceRanker = evidenceRanker
        self.labelEmbedding = labelEmbedding
        self.labelDomains = labelDomains
        self.labelMaskDomain = labelMaskDomain
        self.softmax = torch.nn.Softmax(dim=0).to(self.device)
        self.domain = domain
        self.domainWeights = domainWeights[domain].to(self.device)
        self.domainWeights /= self.domainWeights.max().to(self.device)
        self.domainWeights = F.normalize(self.domainWeights, p=0, dim=0).to(self.device)
        self.domainWeights = self.domainWeights * (1 / torch.sum(self.domainWeights)).to(self.device)
        self.claimDate = nn.Embedding(2, 768).to(self.device)
        self.evidenceDate = nn.Embedding(22, 768).to(self.device)
        self.positionEmbeddings = nn.Embedding(200, 768).to(self.device)
        self.predicateEmbeddings = nn.Embedding(2, 768).to(self.device)
        self.verschil = nn.Embedding(86, 768).to(self.device)
        self.wordEmbeddings = self.transformer.embeddings.word_embeddings.requires_grad_(True)

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),min=1e-9)

    def forward(self,claim,evidences,metadata_encoding,domain,claimDate,snippetDates,verbsClaim,timeExpressionsClaim,positionClaim,timeRefsClaim,
                timeHeidelClaim,verbsSnippets,timeExpressionsSnippets,positionSnippets,timeRefsSnippets,timeHeidelSnippets):
        encoded_input = self.tokenizer(claim, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
        model_output = self.transformer(**encoded_input)
        # Perform pooling
        claim_encoding = self.mean_pooling(model_output, encoded_input['attention_mask']).to(self.device)
        #mask = torch.cat((torch.tensor([1]).to(self.device),encoded_input['attention_mask'].to(self.device).squeeze(0)),dim=0).unsqueeze(0)
        #inputForward = torch.cat((self.claimDate(torch.tensor([claimDate]).to(self.device)),inputForward.squeeze(0)),dim=0)
        #print(inputForward)
        #print(inputForward[0])
        #print(positionClaim)
        #print(positionSnippets)
        positions = positionClaim.split('\t')
        verbs = verbsClaim.split('\t')
        times = timeExpressionsClaim.split('\t')
        verschillenIndices = timeRefsClaim.split('\t')
        verschillenValues = timeHeidelClaim.split('\t')
        tijdAbsolute = torch.zeros(768).to(self.device)
        number = 0
        if verschillenIndices[0] != "":
            for i in range(len(verschillenIndices)):
                if verschillenValues[i].find('Duur') == -1 and verschillenValues[i].find('Refs') == -1:
                    if verschillenValues[i].isdigit():
                        tijdAbsolute = self.verschil(torch.tensor([int(verschillenValues[i])]).squeeze(0).to(self.device))
                        number += 1
                if i + 1 >= len(verschillenValues):
                    break
        if number > 0:
            claim_encoding = 0.1*claim_encoding + 0.9*tijdAbsolute/number
        #claim_encoding = F.normalize(claim_encoding,p=2,dim=0).to(self.device)
        #claim_encoding = (claim_encoding-torch.min(claim_encoding))/(torch.max(claim_encoding)-torch.min(claim_encoding))
        distribution = torch.zeros(len(self.labelDomains[domain])).to(self.device)
        evidences = evidences.split(' 0123456789 ')[:-1]
        verbsSnippets = verbsSnippets.split(' 0123456789 ')[:-1]
        timeExpressionsSnippets = timeExpressionsSnippets.split(' 0123456789 ')[:-1]
        positionSnippets = positionSnippets.split(' 0123456789 ')[:-1]
        timeRefsSnippets = timeRefsSnippets.split(' 0123456789 ')[:-1]
        timeHeidelSnippets = timeHeidelSnippets.split(' 0123456789 ')[:-1]
        snippetDates = snippetDates.split('\t')
        for i in range(len(evidences)):
            encoded_input = self.tokenizer(evidences[i], padding=True, truncation=True, return_tensors='pt',
                                           max_length=512).to(self.device)
            model_output = self.transformer(**encoded_input)
            evidence_encoding = self.mean_pooling(model_output, encoded_input['attention_mask']).to(self.device)
            #print(positionSnippets)
            positionSnippet = positionSnippets[i].split('\t')
            verbsSnippet = verbsSnippets[i].split('\t')
            timeExpressionsSnippet = timeExpressionsSnippets[i].split('\t')
            timeRefsSnippet = timeRefsSnippets[i].split('\t')
            timeHeidelSnippet = timeHeidelSnippets[i].split('\t')
            tijdAbsolute = torch.zeros(768).to(self.device)
            number = 0
            if timeRefsSnippet[0] != "":
                for j in range(len(timeRefsSnippet)):
                    index = timeRefsSnippet[j]
                    if timeHeidelSnippet[j].find('Duur') == -1 and timeHeidelSnippet[j].find('Refs') == -1:
                        if timeHeidelSnippet[j].isdigit():
                            if int(index)<512:
                                tijdAbsolute += self.verschil(
                                    torch.tensor([int(timeHeidelSnippet[j])]).to(self.device)).squeeze(
                                    0).to(self.device)
                                number += 1
                    if j + 1 >= len(timeHeidelSnippet):
                        break
            if number > 0:
                evidence_encoding = 0.1*evidence_encoding+0.9*tijdAbsolute/number
            instance_encoding = self.instanceEncoder(claim_encoding.squeeze(0),evidence_encoding.squeeze(0),metadata_encoding.squeeze(0)).to(self.device)
            rank_evidence = self.evidenceRanker(instance_encoding).to(self.device)
            label_distribution = self.labelEmbedding(instance_encoding,domain).to(self.device)
            label_distributionDomain = self.labelMaskDomain(label_distribution).to(self.device)
            distribution = distribution + torch.mul(rank_evidence,label_distributionDomain).to(self.device)
        return distribution.to(self.device)

    '''
    Function for saving the neural network
    '''
    def saving_NeuralNetwork(model):
        pathI = "/scratch/leuven/347/vsc34763/finalResults/iteratie2/modelsBERTAbsoluteAdamWAdding10A/" + model.domain
        #pathI = "modelsBERTAbsoluteAdamWAdding/" + model.domain
        if not os.path.exists(pathI):
            os.mkdir(pathI)
        #model.metaDataEncoder.saving_NeuralNetwork(pathI + '/encoderMetadata')
        #model.instanceEncoder.saving_NeuralNetwork(pathI + '/instanceEncoder')
        #model.evidenceRanker.saving_NeuralNetwork(pathI+ '/evidenceRanker')
        #model.labelMaskDomain.saving_NeuralNetwork(pathI + '/labelMaskDomain')
        #model.labelEmbedding.saving_NeuralNetwork(pathI + '/labelEmbedding')
        torch.save(model.state_dict(), pathI + '/model')
'''
Function for loading the configurations from a file
It first reads the configurations of a file
Then it initialises a neural network with the parameters of the file
Then it sets the neural network on the state of the loaded neural network
'''
def loading_NeuralNetwork(model):
    pathI ="/scratch/leuven/347/vsc34763/finalResults/iteratie2/modelsBERTAbsoluteAdamWAdding10A/"+model.domain
    #pathI = "modelsBERTAbsoluteAdamWAdding/" + model.domain
    #model.claimEncoder = model.claimEncoder.loading_NeuralNetwork(pathI+'/claimEncoder')
    #model.evidenceEncoder = model.evidenceEncoder.loading_NeuralNetwork(pathI + '/evidenceEncoder')
    #model.metaDataEncoder.loading_NeuralNetwork(pathI + '/encoderMetadata')
    #model.instanceEncoder.loading_NeuralNetwork(pathI + '/instanceEncoder')
    #model.evidenceRanker.loading_NeuralNetwork(pathI + '/evidenceRanker')
    #model.labelMaskDomain.loading_NeuralNetwork(pathI + '/labelMaskDomain')
    #model.labelEmbedding.loading_NeuralNetwork(pathI + '/labelEmbedding')
    model.load_state_dict(torch.load(pathI + '/model'))
    model.eval()
    return model

def eval_loop(dataloader, model,oneHotEncoder,domainLabels,domainLabelIndices,device):
    groundTruthLabels = []
    predictedLabels = []
    #loss = nn.CrossEntropyLoss(model.domainWeights,reduction='sum')
    loss = nn.CrossEntropyLoss(reduction="sum")
    totalLoss = 0
    # Bert transformer for embedding the word captions
    #transformer = SentenceTransformer('paraphrase-distilroberta-base-v1')
    with torch.no_grad():
        for c in dataloader:
            # Compute prediction and loss
            # outcome of feedforwarding feature vector to image neural network
            # set gradient to true
            predictionBatch = torch.tensor([])
            for i in range(len(c[0])):
                metaDataClaim = oneHotEncoder.encode(c[3][i],device)
                metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                domain = c[0][0].split('-')[0]
                #print(c[0][i])
                prediction = model(c[1][i], c[2][i], metadata_encoding, domain, c[5][i], c[6][i],
                                   c[7][i],
                                   c[8][i], c[9][i], c[10][i], c[11][i], c[12][i], c[13][i],
                                   c[14][i], c[15][i], c[16][i]).to(device)
                if predictionBatch.size()[0]==0:
                    predictionBatch = prediction.unsqueeze(0).to(device)
                    predictedLabels.append(torch.argmax(prediction).item())
                else:
                    predictionBatch = torch.cat((predictionBatch,prediction.unsqueeze(0))).to(device)
                    predictedLabels.append(torch.argmax(prediction).item())

            labelIndices = []
            for label in c[4]:
                labelIndices.append(domainLabelIndices[domain][domainLabels[domain].index(label)])
                groundTruthLabels.append(domainLabelIndices[domain][domainLabels[domain].index(label)])
            target = torch.tensor(labelIndices).to(device)
            output = loss(predictionBatch,target)
            totalLoss += output.item()


    macro = f1_score(groundTruthLabels, predictedLabels, average='macro')
    micro = f1_score(groundTruthLabels, predictedLabels, average='micro')

    return totalLoss,micro,macro

def getLabelIndicesDomain(domainPath,labelPath,weightsPath):
    domainsIndices = dict()
    domainsLabels = dict()
    domainLabelIndices = dict()
    domainWeights = dict()
    labelSequence = []
    file = open(labelPath,'r')
    lines = file.readlines()
    for line in lines:
        labelSequence.append(line.replace('\n',''))
    file = open(domainPath,'r')
    lines = file.readlines()
    for line in lines:
        parts = line.split("\t")
        labelsDomain = parts[1].split(",")
        labelsDomain[-1] = labelsDomain[-1].replace('\n','')
        labelIndices = []
        for label in labelsDomain:
            labelIndices.append(labelSequence.index(label.replace('\n','')))
        labelIndicesDomainM = sorted(labelIndices)
        labelIndicesDomain = []
        for index in labelIndices:
            labelIndicesDomain.append(labelIndicesDomainM.index(index))
        domainsIndices[parts[0]] = labelIndices
        domainsLabels[parts[0]] = labelsDomain
        domainLabelIndices[parts[0]] = labelIndicesDomain
    file = open(weightsPath, 'r')
    lines = file.readlines()
    for line in lines:

        parts = line.split("\t")
        weightsDomainNormal = parts[1:]
        weightsDomainNormal[-1] = weightsDomainNormal[-1].replace('\n','')
        domainWeights[parts[0]] = torch.zeros(len(weightsDomainNormal))
        for i in range(len(weightsDomainNormal)):
            domainWeights[parts[0]][domainLabelIndices[parts[0]][i]] = float(weightsDomainNormal[i])
    #print(domainWeights)
    return domainsIndices,domainsLabels,domainLabelIndices,domainWeights

def calculatePrecisionDev(dataloader, model,oneHotEncoder,domainLabels,domainLabelIndices,device):
    groundTruthLabels = []
    predictedLabels = []
    with torch.no_grad():
        for batch in dataloader:
            for i in range(len(batch[0])):
                metaDataClaim = oneHotEncoder.encode(batch[3][i],device)
                metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                domain = batch[0][0].split('-')[0]
                #print(c[0][i])
                prediction = model(batch[1][i], batch[2][i], metadata_encoding, domain, batch[5][i], batch[6][i],
                                   batch[7][i],
                                   batch[8][i], batch[9][i], batch[10][i], batch[11][i], batch[12][i], batch[13][i],
                                   batch[14][i], batch[15][i], batch[16][i]).to(device)
                groundTruthLabels.append(domainLabelIndices[domain][domainLabels[domain].index(batch[4][i])])
                predictedLabels.append(torch.argmax(prediction).item())

    print('Micro F1 - score')
    print(f1_score(groundTruthLabels, predictedLabels, average='micro'))
    print('Macro F1-score')
    print(f1_score(groundTruthLabels, predictedLabels, average='macro'))

def getPredictions(dataloader, model,oneHotEncoder,domainLabels,domainLabelIndices,device):
    predictions = dict()
    with torch.no_grad():
        for batch in dataloader:
            for i in range(len(batch[0])):
                metaDataClaim = oneHotEncoder.encode(batch[3][i],device)
                metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                domain = batch[0][0].split('-')[0]
                prediction = model(batch[1][i], batch[2][i], metadata_encoding, domain, batch[5][i], batch[6][i],
                                   batch[7][i],
                                   batch[8][i], batch[9][i], batch[10][i], batch[11][i], batch[12][i], batch[13][i],
                                   batch[14][i], batch[15][i], batch[16][i]).to(device)
                pIndex = torch.argmax(prediction).item()
                plabel = domainLabels[domain][domainLabelIndices[domain].index(pIndex)]
                predictions[batch[0][i]] = plabel
    return predictions

def train(batch,model,oneHotEncoder, optimizer,domainLabels,domainLabelIndices,device,preprocessing = False):
    model.train()
    loss = nn.CrossEntropyLoss(reduction="sum")
    '''
    if preprocessing:
        loss = nn.CrossEntropyLoss(reduction='sum')
    else:
        loss = nn.CrossEntropyLoss(model.domainWeights,reduction='sum')
    '''
    predictionBatch = torch.tensor([],requires_grad=True).to(device)
    for i in range(len(batch[0])):
        metaDataClaim = oneHotEncoder.encode(batch[3][i],device).to(device)
        metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
        domain = batch[0][0].split('-')[0]
        prediction = model(batch[1][i], batch[2][i], metadata_encoding, domain,batch[5][i],batch[6][i],batch[7][i],
                           batch[8][i],batch[9][i],batch[10][i],batch[11][i],batch[12][i],batch[13][i],
                           batch[14][i],batch[15][i],batch[16][i]).to(device)
        if predictionBatch.size()[0] == 0:
            predictionBatch = prediction.unsqueeze(0).to(device)
        else:
            predictionBatch = torch.cat((predictionBatch, prediction.unsqueeze(0))).to(device)
    labelIndices = []
    for label in batch[4]:
        labelIndices.append(domainLabelIndices[domain][domainLabels[domain].index(label)])
    target = torch.tensor(labelIndices).to(device)
    output = loss(predictionBatch, target)
    # Backpropagation
    # reset the gradients of model parameters
    optimizer.zero_grad()
    # calculate gradients of the loss w.r.t. each parameter
    output.backward()
    #  adjust the parameters by the gradients collected in the backward pass
    optimizer.step()
    return output.item()


def writePredictions(predictions,file,output):
    with open(file,'r',encoding='utf-8') as file:
        with open(output,'w',encoding='utf-8') as output:
            for line in file:
                elements = line.split('\t')
                output.write(predictions[elements[0]])
                output.write('\n')
            output.close()

def writeMetadata(metadataSet):
    with open('../Metadata_sequence/metadata', 'w', encoding='utf-8') as file:
        for metadata in metadataSet:
            file.write(metadata)
            file.write('\n')
        file.close()

def readMetadata():
    metadataSequence = []
    with open('../Metadata_sequence/metadata', 'r', encoding='utf-8') as file:
        for metadata in file:
            metadataSequence.append(metadata)
        file.close()
    return metadataSequence

def preprocessing(models,epochs=800):
    with torch.no_grad():
        for model in models:
            model[1].eval()
            validation_loss, microF1Test, macroF1Test = eval_loop(model[11], model[1], oneHotEncoderM, domainLabels,
                                                          domainLabelIndices, device)
            model[8](validation_loss, microF1Test, macroF1Test, model[1])
            file = open('/scratch/leuven/347/vsc34763/finalResults/iteratie2/lossAbsoluteBertAdamWAdding10A/' + 'Loss' + model[3], 'a', encoding='utf-8')
            #file = open('lossAbsoluteBertAdamWAdding/' + 'Loss' + model[3], 'a',
            #            encoding='utf-8')
            file.write("Preprocessing stap " + str(0) + " - " + str(validation_loss) + "\n")
            file.close()
            '''
            validation_lossT, microF1TestT, macroF1TestT = eval_loop(model[7], model[1], oneHotEncoderM, domainLabels,
                                                                  domainLabelIndices, device)
            
            file = open('/scratch/leuven/347/vsc34763/lossAbsoluteBertAdamWAdding1/' + 'LossTrain' + model[3], 'a', encoding='utf-8')
            #file = open('lossAbsoluteBertAdamWAdding/' + 'Loss' + model[3], 'a',
            #            encoding='utf-8')
            file.write("Preprocessing stap " + str(0) + " - " + str(validation_lossT) + "\n")
            file.close()
            '''
            if not model[8].early_stop:
                if model[8].better:
                    file = open('/scratch/leuven/347/vsc34763/finalResults/iteratie2/epochsAbsoluteBertAdamWAdding10A/' + model[3], 'w', encoding='utf-8')
                    #file = open('epochsAbsoluteBertAdamWAdding/' + model[3], 'w',
                    #            encoding='utf-8')
                    file.write('Micro - ' + str(microF1Test) + '\n')
                    file.write('Macro - ' + str(macroF1Test) + '\n')
                    file.write(model[8].lastResult + '\n')
                    file.write('Preprocessing')
                    file.close()
                    print(model[3])
                    print('Micro F1-score')
                    print(microF1Test)
                    print('Macro F1 - score')
                    print(macroF1Test)
            else:
                print("Early stopping - Test")
    print('Start preprocessing')
    for i in range(epochs):
        print('Epoch ' + str(i))
        newModels = list()
        for model in models:
            batch = next(model[0])
            # optimizer = torch.optim.RMSprop(model[1].parameters(), lr=2e-4)
            loss = train(batch, model[1], oneHotEncoderM, model[2], domainLabels, domainLabelIndices, device, True)
            model[12].step()
        with torch.no_grad():
            for model in models:
                it = i
                model[1].eval()
                '''
                if i >=50:
                    model[12].step()
                '''
                newModels.append(
                    tuple(
                        [iter(model[7]), model[1], model[2], model[3], model[4], model[5], it, model[7], model[8],
                         model[9],
                         model[10], model[11],model[12]]))
                validation_loss, microF1Test, macroF1Test = eval_loop(model[11], model[1], oneHotEncoderM, domainLabels,
                                                                      domainLabelIndices, device)
                model[8](validation_loss, microF1Test, macroF1Test, model[1])
                file = open('/scratch/leuven/347/vsc34763/finalResults/iteratie2/lossAbsoluteBertAdamWAdding10A/' + 'Loss' + model[3], 'a', encoding='utf-8')
                file.write("Preprocessing stap " + str(it) + " - " + str(validation_loss) + "\n")
                file.close()
                '''
                validation_lossT, microF1TestT, macroF1TestT = eval_loop(model[7], model[1], oneHotEncoderM,
                                                                         domainLabels,
                                                                         domainLabelIndices, device)
                
                file = open('/scratch/leuven/347/vsc34763/lossAbsoluteBertAdamWAdding/' + 'LossTrain' + model[3], 'a', encoding='utf-8')
                #file = open('lossAbsoluteBertAdamWAdding/' + 'Loss' + model[3], 'a',
                #            encoding='utf-8')
                file.write("Preprocessing stap " + str(it) + " - " + str(validation_lossT) + "\n")
                file.close()
                '''
                if not model[8].early_stop:
                    if model[8].better:
                        file = open('/scratch/leuven/347/vsc34763/finalResults/iteratie2/epochsAbsoluteBertAdamWAdding10A/' + model[3], 'w', encoding='utf-8')
                        file.write('Micro - ' + str(microF1Test) + '\n')
                        file.write('Macro - ' + str(macroF1Test) + '\n')
                        file.write(model[8].lastResult + '\n')
                        file.write('Preprocessing')
                        file.write(str(it))
                        file.close()
                        print(model[3])
                        print('Micro F1-score')
                        print(microF1Test)
                        print('Macro F1 - score')
                        print(macroF1Test)
                else:
                    print("Early stopping - Test")
        models = newModels
        random.shuffle(models)
    #print('Average loss : ' ,  totalLoss/size)

if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    domainIndices,domainLabels,domainLabelIndices,domainWeights = getLabelIndicesDomain('labels/labels.tsv','labels/labelSequence','labels/weights.tsv')
    oneHotEncoderM = oneHotEncoder('Metadata_sequence/metadata')
    domains = {'abbc', 'afck', 'bove', 'chct', 'faan', 'farg', 'para', 'thet', 'thal',
               'vees', 'vogo', 'wast', 'hoer', 'pose'}
    metadataSet = set()
    labelEmbeddingLayerM = labelEmbeddingLayer(2308, domainIndices)
    domainModels = []
    losses = np.zeros(len(domains))
    index = 0
    transformer = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
    encoderMetadataM = encoderMetadata(3, 3, oneHotEncoderM).to(device)
    instanceEncoderM = instanceEncoder().to(device)
    evidenceRankerM = evidenceRanker(2308, 100).to(device)
    for domain in domains:
        dev_set = NUS(mode="Dev", path='dev/dev-' + domain + '.tsv', pathToSave="dev/time/dataset2/", domain=domain)
        train_set = NUS(mode='Train', path='train/train-' + domain + '.tsv', pathToSave="train/time/dataset2/",
                        domain=domain)
        test_set = NUS(mode='Test', path='test/test-' + domain + '.tsv', pathToSave="test/time/dataset2/",
                       domain=domain)
        trainMetadata = train_set.getMetaDataSet()
        devMetadata = dev_set.getMetaDataSet()
        testMetadata = test_set.getMetaDataSet()
        metadataSet = set.union(metadataSet, trainMetadata)
        metadataSet = set.union(metadataSet, devMetadata)
        metadataSet = set.union(metadataSet, testMetadata)
        train_loader = DataLoader(train_set,
                                  batch_size=32,
                                  shuffle=True)
        dev_loader = DataLoader(dev_set,
                                batch_size=32,
                                shuffle=True)
        test_loader = DataLoader(test_set,
                                 batch_size=32,
                                 shuffle=True)
        labelMaskDomainM = labelMaskDomain(2308, domainIndices, domain, len(domainIndices[domain])).to(device)
        # labelSequence, domainsNew = readLabels('labels/labels.tsv', 'labels/labelSequence')

        verificationModelM = verifactionModel(transformer, encoderMetadataM, instanceEncoderM,
                                              evidenceRankerM,
                                              labelEmbeddingLayerM, labelMaskDomainM, domainIndices, domainWeights,
                                              domain).to(device)
        optimizer1 = torch.optim.AdamW(verificationModelM.parameters(), lr=1e-4)
        optimizer2 = torch.optim.AdamW(verificationModelM.parameters(), lr=1e-4)
        domainModel = [train_loader, dev_loader, test_loader, verificationModelM, optimizer1, domain, index, optimizer2]
        domainModels.append(domainModel)
        index += 1
    # writeMetadata(metadataSet)
    # dataloader for the train-set
    # dataloader for the test-set
    # number of epochs
    epochs = 300
    # This is the bestAcc we have till now
    bestAcc = 0
    # retrain = {'afck','bove','chct','clck','faly','farg','hoer','mpws','para','peck','pomt','pose','snes','thal','thet','tron','vees','vogo','wast'}

    models = set()
    with torch.no_grad():
        for model in domainModels:
            # early_stopping
            early_stopping = EarlyStopping(patience=3, verbose=True)
            NNmodel = model[3]
            NNmodel.eval()
            # print("Results model na epochs " + str(epochs) + ' ' + model[5])
            # calculatePrecisionDev(model[1], NNmodel, oneHotEncoderM, domainLabels, domainLabelIndices, device)
            validation_loss, microF1, macroF1 = eval_loop(model[1], NNmodel, oneHotEncoderM, domainLabels,
                                                          domainLabelIndices, device)
            early_stopping(validation_loss, microF1, macroF1, NNmodel)
            optimizer1 = AdamW(NNmodel.parameters(), lr=5e-3)
            numberOfTrainingSteps = 6000
            scheduler = get_linear_schedule_with_warmup(
                optimizer1, num_warmup_steps=0,
                num_training_steps=numberOfTrainingSteps
            )
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.95)
            optimizer2 = optimizer1
            models.add(
                tuple([iter(model[0]), model[3], optimizer1, model[5],
                       model[6], model[2], 0, model[0], early_stopping,
                       optimizer2, model[1], scheduler]))

    # preprocessing(models)

    models = set()
    domainsOptimizer1 = ["hoer", "goop", "pomt", "snes", "tron", "pose"]

    with torch.no_grad():
        for model in domainModels:
            # early_stopping
            early_stopping = EarlyStopping(patience=3, verbose=True)
            NNmodel = loading_NeuralNetwork(model[3])
            NNmodel.eval()
            print("Results model na preprocessing " + ' ' + model[5])
            calculatePrecisionDev(model[1], NNmodel, oneHotEncoderM, domainLabels, domainLabelIndices, device)
            validation_loss, microF1, macroF1 = eval_loop(model[1], NNmodel, oneHotEncoderM, domainLabels,
                                                          domainLabelIndices, device)
            early_stopping(validation_loss, microF1, macroF1, NNmodel)
            optimizer1 = AdamW(NNmodel.parameters(), lr=5e-3)
            numberOfTrainingSteps = 6000
            scheduler = get_linear_schedule_with_warmup(
                optimizer1, num_warmup_steps=0,
                num_training_steps=numberOfTrainingSteps
            )
            if model[5] in domainsOptimizer1:
                optimizer1 = torch.optim.AdamW(NNmodel.parameters(), lr=1e-4)
                numberOfTrainingSteps = len(model[0])
                if numberOfTrainingSteps % 32 == 0:
                    numberOfTrainingSteps = numberOfTrainingSteps / 32 * epochs * 100
                else:
                    numberOfTrainingSteps = (numberOfTrainingSteps // 32 + 1) * epochs * 100
                scheduler = get_linear_schedule_with_warmup(
                    optimizer1, num_warmup_steps=0,
                    num_training_steps=numberOfTrainingSteps
                )
            else:
                optimizer1 = torch.optim.AdamW(NNmodel.parameters(), lr=1e-4, weight_decay=5e-4)
                scheduler = "none"
            optimizer2 = optimizer1

            models.add(
                tuple([iter(model[0]), model[3], optimizer1, model[5],
                       model[6], model[2], 0, model[0], early_stopping,
                       optimizer2, model[1], scheduler]))

    print('start finetuning')
    while models:
        removeModels = []
        removeEntirely = set()
        for model in models:
            batch = next(model[0], "None")
            try:
                if batch != "None":
                    loss = train(batch, model[1], oneHotEncoderM, model[2], domainLabels, domainLabelIndices, device)
                    losses[model[4]] += loss
                    if model[12] != "none":
                        model[12].step()
                else:
                    removeModels.append(model)
            except:
                removeEntirely.add(model)
        if len(removeModels) != 0:
            for model in removeModels:
                models.remove(model)
                if model not in removeEntirely:
                    model[1].eval()
                    it = model[6] + 1
                    validation_loss, microF1Test, macroF1Test = eval_loop(model[11], model[1], oneHotEncoderM,
                                                                          domainLabels,
                                                                          domainLabelIndices, device)
                    model[8](validation_loss, microF1Test, macroF1Test, model[1])
                    file = open(
                        '/scratch/leuven/347/vsc34763/finalResults/iteratie2/lossAbsoluteBertAdamWAdding10A/' + 'Loss' +
                        model[3],
                        'a',
                        encoding='utf-8')
                    file.write("Preprocessing stap " + str(it) + " - " + str(validation_loss) + "\n")
                    file.close()
                    if not model[8].early_stop and it < epochs:
                        if model[8].better:
                            file = open(
                                '/scratch/leuven/347/vsc34763/finalResults/iteratie2/epochsAbsoluteBertAdamWAdding10A/' +
                                model[3],
                                'w', encoding='utf-8')
                            file.write('Micro - ' + str(microF1Test) + '\n')
                            file.write('Macro - ' + str(macroF1Test) + '\n')
                            file.write(model[8].lastResult + '\n')
                            file.write(str(it))
                            file.close()
                            print('Test')
                            print(model[3])
                            print('Micro F1-score')
                            print(microF1Test)
                            print('Macro F1 - score')
                            print(macroF1Test)
                        models.add(tuple(
                            [iter(model[7]), model[1], model[2], model[3], model[4], model[5], it, model[7],
                             model[8],
                             model[9], model[10], model[11], model[12]]))
                    else:
                        print("Early stopping")
                    losses[model[4]] = 0

    precisionModels = dict()
    microF1All = 0
    macroF1All = 0
    for model in domainModels:
        NNmodel = loading_NeuralNetwork(model[3]).to(device)
        print("Results model na epochs " + model[5])
        calculatePrecisionDev(model[2], NNmodel, oneHotEncoderM, domainLabels, domainLabelIndices, device)
        validation_loss, microF1, macroF1 = eval_loop(model[2], NNmodel, oneHotEncoderM, domainLabels,
                                                      domainLabelIndices, device)
        print('Loss - ' + str(validation_loss))
        microF1All += microF1
        macroF1All += macroF1
        # precision = getPredictions(model[2], NNmodel, oneHotEncoderM, domainLabels, domainLabelIndices, device)
        # precisionModels.update(precision)
    print('Average micro ')
    print(str(microF1All / 26))
    print('Average macro')
    print(str(macroF1All / 26))

    writePredictions(precisionModels, "test/test.tsv", "test.predict")