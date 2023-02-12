import os
import sys
import random

import numpy as np
from pytorchtools import EarlyStopping
import torch.nn.functional as F

import torch
from torch.utils.data import DataLoader

from labelMaskDomain import labelMaskDomain
'''
combining pre-created partitions into one dataset
'''
from datasetIteratie1Combiner import dump_load, dump_write, NUS
'''
import this for creating a new dataset
and comment import datasetIteratie1Combiner above
#from datasetIteratie1 import dump_load, dump_write, NUS
'''
from encoderMetadata import encoderMetadata
from evidence_ranker import evidenceRanker
from instanceEncoder import instanceEncoder
from labelEmbeddingLayer import labelEmbeddingLayer
from OneHotEncoder import oneHotEncoder
from torch import nn
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

from transformers import BertTokenizer, AutoModel, AutoTokenizer

'''
Local version of extension 1: adding buckets of publicationtime
With DistilRoBERTa encoder
'''
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
        self.wordEmbeddings = self.transformer.embeddings.word_embeddings.requires_grad_(True)

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),min=1e-9)


    def forward(self,claim,evidences,metadata_encoding,domain,claimDate,snippetDates):
        encoded_input = self.tokenizer(claim, padding=True, truncation=True, return_tensors='pt', max_length=511)
        inputForward = self.wordEmbeddings(encoded_input['input_ids'].to(self.device)).to(self.device)
        mask = torch.cat((torch.tensor([1]).to(self.device),encoded_input['attention_mask'].to(self.device).squeeze(0)),dim=0).unsqueeze(0)
        inputForward = torch.cat((self.claimDate(torch.tensor([claimDate]).to(self.device)),inputForward.squeeze(0)),dim=0)
        # with torch.no_grad():
        model_output = self.transformer(inputs_embeds=inputForward.unsqueeze(0),attention_mask = mask)
        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, mask).to(self.device)

        claim_encoding = F.normalize(sentence_embeddings, p=2, dim=1).to(self.device).squeeze(0)
        distribution = torch.zeros(len(self.labelDomains[domain])).to(self.device)
        evidences = evidences.split(' 0123456789 ')[:-1]
        snippetDates = snippetDates.split('\t')
        for i in range(len(evidences)):
            encoded_input = self.tokenizer(evidences[i], padding=True, truncation=True, return_tensors='pt',
                                           max_length=512)
            inputForward = self.wordEmbeddings(encoded_input['input_ids'].to(self.device)).to(self.device)
            mask = torch.cat((torch.tensor([1]).to(self.device), encoded_input['attention_mask'].to(self.device).squeeze(0)),
                             dim=0).unsqueeze(0)
            inputForward = torch.cat(
                (self.evidenceDate(torch.tensor([int(snippetDates[i+1])]).to(self.device)), inputForward.squeeze(0)), dim=0)
            # with torch.no_grad():
            #input = dict()
            #input['inputs_embeds'] = inputForward.unsqueeze(0)
            #input['attention_mask'] = mask
            model_output = self.transformer(inputs_embeds=inputForward.unsqueeze(0),attention_mask = mask)
            # Perform pooling
            sentence_embeddings = self.mean_pooling(model_output, mask).to(self.device)
            evidence_encoding = F.normalize(sentence_embeddings, p=2, dim=1).to(self.device).squeeze(0)+ self.evidenceDate(torch.tensor([int(snippetDates[i+1])]).to(self.device)).squeeze(0).to(self.device)
            instance_encoding = self.instanceEncoder(claim_encoding.squeeze(0),evidence_encoding.squeeze(0),metadata_encoding.squeeze(0)).to(self.device)
            rank_evidence = self.evidenceRanker(instance_encoding).to(self.device)
            label_distribution = self.labelEmbedding(instance_encoding,domain).to(self.device)
            label_distributionDomain = self.labelMaskDomain(label_distribution).to(self.device)
            distribution = distribution + torch.mul(rank_evidence,label_distributionDomain).to(self.device)
        return distribution.to(self.device)

    '''
       Function for saving the neural network
   '''
    def saving_NeuralNetwork(model, path):
        pathI = path + "/" + model.domain
        if not os.path.exists(pathI):
            os.mkdir(pathI)
        torch.save(model.state_dict(), pathI + '/model')

    '''
    Function for loading the neural network
    '''
    def loading_NeuralNetwork(model, path):
        pathI = path + "/" + model.domain
        model.load_state_dict(torch.load(pathI + '/model', map_location=torch.device('cpu')))
        model.eval()
        return model

def eval_loop(dataloader, model,oneHotEncoder,domainLabels,domainLabelIndices,device):
    groundTruthLabels = []
    predictedLabels = []
    loss = nn.CrossEntropyLoss(reduction='sum')
    totalLoss = 0
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
                prediction= model(c[1][i],c[2][i],metadata_encoding,domain,c[5][i],c[6][i]).to(device)
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
    return domainsIndices,domainsLabels,domainLabelIndices,domainWeights

def calculatePrecisionDev(dataloader, model,oneHotEncoder,domainLabels,domainLabelIndices,device):
    groundTruthLabels = []
    predictedLabels = []
    with torch.no_grad():
        for c in dataloader:
            for i in range(len(c[0])):
                metaDataClaim = oneHotEncoder.encode(c[3][i],device)
                metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                domain = c[0][0].split('-')[0]
                prediction = model(c[1][i], c[2][i], metadata_encoding, domain,c[5][i],c[6][i]).to(device)
                groundTruthLabels.append(domainLabelIndices[domain][domainLabels[domain].index(c[4][i])])
                predictedLabels.append(torch.argmax(prediction).item())

    print('Micro F1 - score')
    print(f1_score(groundTruthLabels, predictedLabels, average='micro'))
    print('Macro F1-score')
    print(f1_score(groundTruthLabels, predictedLabels, average='macro'))

def getPredictions(dataloader, model,oneHotEncoder,domainLabels,domainLabelIndices,device):
    predictions = dict()
    with torch.no_grad():
        for c in dataloader:
            for i in range(len(c[0])):
                metaDataClaim = oneHotEncoder.encode(c[3][i],device)
                metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                domain = c[0][0].split('-')[0]
                prediction = model(c[1][i], c[2][i], metadata_encoding, domain,batch[5][i],batch[6][i]).to(device)
                pIndex = torch.argmax(prediction).item()
                plabel = domainLabels[domain][domainLabelIndices[domain].index(pIndex)]
                predictions[c[0][i]] = plabel
    return predictions

def train(batch,model,oneHotEncoder, optimizer,domainLabels,domainLabelIndices,device,preprocessing = False):
    model.train()
    loss = nn.CrossEntropyLoss(reduction='sum')
    predictionBatch = torch.tensor([],requires_grad=True).to(device)
    for i in range(len(batch[0])):
        metaDataClaim = oneHotEncoder.encode(batch[3][i],device).to(device)
        metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
        domain = batch[0][0].split('-')[0]
        prediction = model(batch[1][i], batch[2][i], metadata_encoding, domain,batch[5][i],batch[6][i]).to(device)
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
    with open('Metadata_sequence/metadata', 'w', encoding='utf-8') as file:
        for metadata in metadataSet:
            file.write(metadata)
            file.write('\n')
        file.close()

def readMetadata():
    metadataSequence = []
    with open('Metadata_sequence/metadata', 'r', encoding='utf-8') as file:
        for metadata in file:
            metadataSequence.append(metadata)
        file.close()
    return metadataSequence

def preprocessing(models,epochs=300):
    with torch.no_grad():
        for model in models:
            model[1].eval()
            validation_loss, microF1Test, macroF1Test = eval_loop(model[9], model[1], oneHotEncoderM, domainLabels,
                                                          domainLabelIndices, device)
            model[8](validation_loss, microF1Test, macroF1Test, model[1])
    print('Start preprocessing')
    for i in range(epochs):
        print('Epoch ' + str(i))
        newModels = list()
        for model in models:
            batch = next(model[0])
            loss = train(batch,model[1],oneHotEncoderM,model[2],domainLabels,domainLabelIndices,device,True)
        with torch.no_grad():
            for model in models:
                it = i
                model[1].eval()
                newModels.append(
                    tuple(
                        [iter(model[7]), model[1], model[2], model[3], model[4], model[5], it, model[7], model[8],
                         model[9],
                         ]))
                validation_loss, microF1Test, macroF1Test = eval_loop(model[9], model[1], oneHotEncoderM, domainLabels,
                                                                      domainLabelIndices, device)
                model[8](validation_loss, microF1Test, macroF1Test, model[1])
        models = newModels
        random.shuffle(models)

if __name__ == "__main__":
    '''
        argument 1 path to save the model/where previous model is saved
                 2 path to save if you construct a new dataset
    '''
    torch.manual_seed(1)
    random.seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    domainIndices, domainLabels, domainLabelIndices, domainWeights = getLabelIndicesDomain(
        os.pardir + '/labels/labels.tsv', os.pardir + '/labels/labelSequence', os.pardir + '/labels/weights.tsv')
    oneHotEncoderM = oneHotEncoder('Metadata_sequence/metadata')
    domains = domainIndices.keys()
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
        '''
        construct a new dataset
        train_set = NUS(mode='Train', path=os.pardir + '/train/train-' + domain + '.tsv', domain=domain,
                        pathToSave=sys.argv[2], number=1)
        dev_set = NUS(mode='Dev', path=os.pardir + '/dev/dev-' + domain + '.tsv', domain=domain,
                        pathToSave=sys.argv[2], number=1)
        test_set = NUS(mode='Test', path=os.pardir + '/test/test-' + domain + '.tsv', domain=domain,
                        pathToSave=sys.argv[2], number=1)
        '''
        '''
        Combine presaved partitions of the dataset into one dataset
        '''
        dev_set = NUS(mode="Dev", path=os.pardir + '/dev/dev-' + domain + '.tsv',
                      pathToSave=os.pardir + "/dev/time/dataset2/", domain=domain)
        train_set = NUS(mode='Train', path=os.pardir + '/train/train-' + domain + '.tsv',
                        pathToSave=os.pardir + "/train/time/dataset2/",
                        domain=domain)
        test_set = NUS(mode='Test', path=os.pardir + '/test/test-' + domain + '.tsv',
                       pathToSave=os.pardir + "/test/time/dataset2/",
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
        labelMaskDomainM = labelMaskDomain(2308,domainIndices,domain,len(domainIndices[domain])).to(device)

        verificationModelM = verifactionModel(transformer,encoderMetadataM, instanceEncoderM,
                                            evidenceRankerM,
                                            labelEmbeddingLayerM,labelMaskDomainM, domainIndices,domainWeights,domain).to(device)
        domainModel = [train_loader, dev_loader, test_loader, verificationModelM, domain, index]
        domainModels.append(domainModel)
        index += 1

    if sys.argv[2] == "evaluation":
        precisionModels = dict()
        microF1All = 0
        macroF1All = 0
        for model in domainModels:
            NNmodel = model[3].loading_NeuralNetwork(sys.argv[1]).to(device)
            print("Results model na epochs " + model[4])
            calculatePrecisionDev(model[2], NNmodel, oneHotEncoderM, domainLabels, domainLabelIndices, device)
            validation_loss, microF1, macroF1 = eval_loop(model[2], NNmodel, oneHotEncoderM, domainLabels,
                                                          domainLabelIndices, device)
            print('Loss - ' + str(validation_loss))
            microF1All += microF1
            macroF1All += macroF1
        print('Average micro ')
        print(str(microF1All / 26))
        print('Average macro')
        print(str(macroF1All / 26))
    else:
        # number of epochs
        epochs = 300

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
                optimizer = torch.optim.AdamW(NNmodel.parameters(), lr=1e-4)
                models.add(
                    tuple([iter(model[0]), model[3], optimizer, model[4],
                           model[5], model[2], 0, model[0], early_stopping,
                           model[1]]))

        preprocessing(models)

        models = set()

        with torch.no_grad():
            for model in domainModels:
                # early_stopping
                early_stopping = EarlyStopping(patience=3, verbose=True)
                NNmodel = model[3].loading_NeuralNetwork(sys.argv[1]).to(device)
                NNmodel.eval()
                print("Results model na preprocessing " + ' ' + model[4])
                calculatePrecisionDev(model[1], NNmodel, oneHotEncoderM, domainLabels, domainLabelIndices, device)
                validation_loss, microF1, macroF1 = eval_loop(model[1], NNmodel, oneHotEncoderM, domainLabels,
                                                              domainLabelIndices, device)
                early_stopping(validation_loss, microF1, macroF1, NNmodel)
                optimizer = torch.optim.AdamW(NNmodel.parameters(), lr=1e-4)
                models.add(
                    tuple([iter(model[0]), model[3], optimizer, model[4],
                           model[5], model[2], 0, model[0], early_stopping,
                           model[1]]))

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
                        validation_loss, microF1Test, macroF1Test = eval_loop(model[9], model[1], oneHotEncoderM,
                                                                              domainLabels,
                                                                              domainLabelIndices, device)
                        model[8](validation_loss, microF1Test, macroF1Test, model[1])
                        if not model[8].early_stop and it < epochs:
                            models.add(tuple(
                                [iter(model[7]), model[1], model[2], model[3], model[4], model[5], it, model[7],
                                 model[8],
                                 model[9]]))
                        else:
                            print("Early stopping")
                        losses[model[4]] = 0
        precisionModels = dict()
        microF1All = 0
        macroF1All = 0
        for model in domainModels:
            NNmodel = model[3].loading_NeuralNetwork(sys.argv[1]).to(device)
            print("Results model na epochs " + model[4])
            calculatePrecisionDev(model[2], NNmodel, oneHotEncoderM, domainLabels, domainLabelIndices, device)
            validation_loss, microF1, macroF1 = eval_loop(model[2], NNmodel, oneHotEncoderM, domainLabels,
                                                          domainLabelIndices, device)
            print('Loss - ' + str(validation_loss))
            microF1All += microF1
            macroF1All += macroF1
        print('Average micro ')
        print(str(microF1All / 26))
        print('Average macro')
        print(str(macroF1All / 26))