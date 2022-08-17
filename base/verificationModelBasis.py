import os
import pickle
import sys
import random

import numpy as np
from base.pytorchtools import EarlyStopping

import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

from base.labelMaskDomainBasis import labelMaskDomain
from base.dataset import dump_load, dump_write, NUS
from base.encoderBasis import encoder
from base.encoderMetadataBasis import encoderMetadata
from base.evidence_rankerBasis import evidenceRanker
from base.instanceEncoderBasis import instanceEncoder
from base.labelEmbeddingLayerBasis import labelEmbeddingLayer
from base.OneHotEncoderBasis import oneHotEncoder
from torch import nn
from sklearn.metrics import f1_score

from transformers import AutoTokenizer, get_linear_schedule_with_warmup


class verifactionModel(nn.Module):
    # Create neural network
    def __init__(self,encoder,metadataEncoder,instanceEncoder,evidenceRanker,labelEmbedding,labelMaskDomain,labelDomains,domainWeights,domain):
        super(verifactionModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder  = encoder
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
        self.domainWeights = F.normalize(self.domainWeights,p=0,dim=0).to(self.device)
        self.domainWeights = self.domainWeights*(1/torch.sum(self.domainWeights)).to(self.device)

    def forward(self,claim,evidences,metadata_encoding,domain):
        claim_encoding = self.encoder(claim).to(self.device)
        distribution = torch.zeros(len(self.labelDomains[domain]),requires_grad=True).to(self.device)
        evidences = evidences.split(' 0123456789 ')[:-1]
        for snippet in evidences:
            evidence_encoding = self.encoder(snippet).to(self.device)
            instance_encoding = self.instanceEncoder(claim_encoding,evidence_encoding,metadata_encoding.squeeze(0)).to(self.device)
            rank_evidence = self.evidenceRanker(instance_encoding).to(self.device)
            label_distribution = self.labelEmbedding(instance_encoding,domain).to(self.device)
            label_distributionDomain = self.labelMaskDomain(label_distribution).to(self.device)
            distribution = distribution + torch.mul(rank_evidence,label_distributionDomain).to(self.device)
        return distribution.to(self.device)

    def forwardAttribution(self, claim, evidences, metadata_encoding, domain, claimDate, snippetDates):
        claim_encoding = self.encoder.forward(claim)
        distribution = torch.zeros(len(self.labelDomains[self.domain])).to(self.device)
        evidences = evidences.split(' 0123456789 ')[:-1]
        evidenceEncodings = []
        for i in range(len(evidences)):
            evidence_encoding = self.encoder.forward(
                evidences[i])
            evidenceEncodings.append(evidence_encoding)
            instance_encoding = self.instanceEncoder(claim_encoding, evidence_encoding,
                                                     metadata_encoding.squeeze(0)).to(self.device)
            rank_evidence = self.evidenceRanker(instance_encoding).to(self.device)
            label_distribution = self.labelEmbedding(instance_encoding, self.domain).to(self.device)
            label_distributionDomain = self.labelMaskDomain(label_distribution).to(self.device)
            distribution = distribution + torch.mul(rank_evidence, label_distributionDomain).to(self.device)
        return distribution.to(self.device), claim_encoding, evidenceEncodings

    def forwardIntegrated(self, inputs, metadata_encoding):
        claimEncoding = inputs[0]
        distribution = torch.zeros(len(self.labelDomains[self.domain])).to(self.device)
        for i in range(len(inputs[1])):
            evidence_encoding = inputs[1][i]
            instance_encoding = self.instanceEncoder(claimEncoding, evidence_encoding,
                                                     metadata_encoding.squeeze(0)).to(self.device)
            rank_evidence = self.evidenceRanker(instance_encoding).to(self.device)
            label_distribution = self.labelEmbedding(instance_encoding, self.domain).to(self.device)
            label_distributionDomain = self.labelMaskDomain(label_distribution).to(self.device)
            distribution = distribution + torch.mul(rank_evidence, label_distributionDomain).to(self.device)
        return distribution.to(self.device)


    def getBaseLine(self, encodingClaim, encodingEvidence, metadata):
        with torch.no_grad():
            instanceEncoding = self.instanceEncoder(encodingClaim, encodingEvidence, metadata)
            rank_evidence = self.evidenceRanker(instanceEncoding)
            label_distribution = self.labelEmbedding(instanceEncoding, self.domain).to(self.device)
            label_distributionDomain = self.labelMaskDomain(label_distribution).to(self.device)

            distribution = torch.mul(rank_evidence, label_distributionDomain).to(self.device)
            distribution = torch.abs(distribution)
            differenceToZero = torch.max(distribution).item()
            bestClaimEncoding = encodingClaim
            bestEvicenceEncoding = encodingEvidence
            low, high = -0.25, 0.25
            while (abs(differenceToZero) > 1e-5):
                encodingClaim = torch.distributions.uniform.Uniform(low, high).sample([256])
                encodingEvidence = torch.distributions.uniform.Uniform(low, high).sample([256])
                instanceEncoding = self.instanceEncoder(encodingClaim, encodingEvidence, metadata)
                rank_evidence = self.evidenceRanker(instanceEncoding)
                label_distribution = self.labelEmbedding(instanceEncoding, self.domain).to(self.device)
                label_distributionDomain = self.labelMaskDomain(label_distribution).to(self.device)

                distribution = torch.mul(rank_evidence, label_distributionDomain).to(self.device)
                distribution = torch.abs(distribution)
                if torch.max(distribution).item() < differenceToZero:
                    differenceToZero = torch.max(distribution).item()
                    bestClaimEncoding = encodingClaim
                    bestEvicenceEncoding = encodingEvidence
                    torch.save(bestClaimEncoding, 'baselineClaimV.pt')
                    torch.save(bestEvicenceEncoding, 'baselineEvidenceV.pt')
                    print(differenceToZero)
            print("lowest result")
            print(differenceToZero)
            print(distribution)
            return bestClaimEncoding, bestEvicenceEncoding

    def getRankingEvidencesLabels(self, claim, evidences, metadata_encoding, domain):
        claim_encoding = self.encoder(claim).to(self.device)
        ranking = []
        labelsAll = []
        labelsDomain = []
        lastInstance_encoding = "None"
        allEqual = True
        evidences = evidences.split(' 0123456789 ')[:-1]
        for snippet in evidences:
            evidence_encoding = self.encoder(snippet).to(self.device)
            instance_encoding = self.instanceEncoder(claim_encoding, evidence_encoding,
                                                     metadata_encoding.squeeze(0)).to(self.device)
            if allEqual:
                if lastInstance_encoding == "None":
                    lastInstance_encoding = instance_encoding
                else:
                    allEqual = torch.equal(lastInstance_encoding, instance_encoding)
                    lastInstance_encoding = instance_encoding
            rank_evidence = self.evidenceRanker(instance_encoding).to(self.device)
            ranking.append(rank_evidence.cpu().detach().item())
            label_distribution = self.labelEmbedding(instance_encoding, domain).to(self.device)
            labelsAll.append(label_distribution.cpu().detach().tolist())
            label_distributionDomain = self.labelMaskDomain(label_distribution).to(self.device)
            labelsDomain.append(label_distributionDomain.cpu().detach().tolist())
        return ranking, labelsAll, labelsDomain, allEqual

    '''
    Function for saving the neural network
    '''
    def saving_NeuralNetwork(model,path):
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(model.state_dict(), path + '/model')

    '''
    Function for loading the configurations from a file
    '''
    def loading_NeuralNetwork(model,path):
        model.load_state_dict(torch.load(path + '/model',map_location=torch.device('cpu')))
        model.eval()
        return model

    def getClaimEncoding(self, claim):
        return self.encoder.getEncoding(claim).to(self.device)

    def getSnippetEncodings(self, evidences, snippetDates):
        evidences = evidences.split(' 0123456789 ')[:-1]
        evidenceEncodings = torch.tensor([])
        for i in range(len(evidences)):
            evidence_encoding = self.encoder.getEncoding(evidences[i]).to(self.device)
            evidenceEncodings = torch.cat((evidenceEncodings, evidence_encoding))
        return evidenceEncodings


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
                prediction= model(c[1][i],c[2][i],metadata_encoding,domain).to(device)
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
                prediction = model(c[1][i], c[2][i], metadata_encoding, domain).to(device)
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
                prediction = model(c[1][i], c[2][i], metadata_encoding, domain).to(device)
                pIndex = torch.argmax(prediction).item()
                plabel = domainLabels[domain][domainLabelIndices[domain].index(pIndex)]
                predictions[c[0][i]] = plabel
    return predictions

def train(batch,model,oneHotEncoder, optimizer,domainLabels,domainLabelIndices,device):
    model.train()
    loss = nn.CrossEntropyLoss(reduction='sum')
    predictionBatch = torch.tensor([],requires_grad=True).to(device)
    for i in range(len(batch[0])):
        metaDataClaim = oneHotEncoder.encode(batch[3][i],device).to(device)
        metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
        domain = batch[0][0].split('-')[0]
        prediction = model(batch[1][i], batch[2][i], metadata_encoding, domain).to(device)
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


def preprocessing(models, epochs=600):
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
            loss = train(batch, model[1], oneHotEncoderM, model[2], domainLabels, domainLabelIndices, device)
            model[10].step()
        with torch.no_grad():
            for model in models:
                it = i
                model[1].eval()
                newModels.append(
                    tuple(
                        [iter(model[7]), model[1], model[2], model[3], model[4], model[5], it, model[7], model[8],
                         model[9],model[10],
                         ]))
                validation_loss, microF1Test, macroF1Test = eval_loop(model[9], model[1], oneHotEncoderM,
                                                                      domainLabels,
                                                                      domainLabelIndices, device)

                model[8](validation_loss, microF1Test, macroF1Test, model[1])

        models = newModels
        random.shuffle(models)

if __name__ == "__main__":
    '''
    argument path to save the model/where previous model is saved
    '''
    torch.manual_seed(1)
    random.seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    domainIndices, domainLabels, domainLabelIndices, domainWeights = getLabelIndicesDomain(os.pardir + '/labels/labels.tsv',
                                                                                           os.pardir + '/labels/labelSequence',
                                                                                           os.pardir + '/labels/weights.tsv')
    oneHotEncoderM = oneHotEncoder('Metadata_sequence/metadata')
    domains = domainIndices.keys()
    metadataSet = set()
    labelEmbeddingLayerM = labelEmbeddingLayer(772, domainIndices)
    domainModels = []
    losses = np.zeros(len(domains))
    index = 0
    encoderM = encoder(300, 128).to(device)
    encoderMetadataM = encoderMetadata(3, 3, oneHotEncoderM).to(device)
    instanceEncoderM = instanceEncoder().to(device)
    evidenceRankerM = evidenceRanker(772, 100).to(device)
    for domain in domains:
        #train_set = dump_load(os.pardir + "/train/base/trainDataset-" + domain)
        #dev_set = dump_load(os.pardir + "/dev/base/devDataset-" + domain)
        #test_set = dump_load(os.pardir + "/test/base/testDataset-" + domain)
        train_set = NUS(mode='Train', path=os.pardir + '/train/train-' + domain + '.tsv', domain=domain)
        dev_set = NUS(mode='Dev', path=os.pardir + '/dev/dev-' + domain + '.tsv', domain=domain)
        test_set = NUS(mode='Test', path=os.pardir + '/test/test-' + domain + '.tsv', domain=domain)
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
        labelMaskDomainM = labelMaskDomain(772, domainIndices, domain, len(domainIndices[domain])).to(device)

        verificationModelM = verifactionModel(encoderM, encoderMetadataM, instanceEncoderM,
                                              evidenceRankerM,
                                              labelEmbeddingLayerM, labelMaskDomainM, domainIndices, domainWeights,
                                              domain).to(device)
        domainModel = [train_loader, dev_loader, test_loader, verificationModelM, domain, index]
        domainModels.append(domainModel)
        index += 1

    # number of epochs
    epochs = 300
    models = set()
    with torch.no_grad():
        for model in domainModels:
            # early_stopping
            early_stopping = EarlyStopping(patience=3, verbose=True)
            NNmodel = model[3]
            NNmodel.eval()

            validation_loss, microF1, macroF1 = eval_loop(model[1], NNmodel, oneHotEncoderM, domainLabels,
                                                          domainLabelIndices, device)
            early_stopping(validation_loss, microF1, macroF1, NNmodel)
            optimizer = torch.optim.Adam(NNmodel.parameters(), lr=1e-3)
            numberOfTrainingSteps = 6000
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0,
                num_training_steps=numberOfTrainingSteps
            )
            models.add(
                tuple([iter(model[0]), model[3], optimizer, model[4],
                       model[5], model[2], 0, model[0], early_stopping,
                    model[1], scheduler]))

    preprocessing(models)

    models = set()

    with torch.no_grad():
        for model in domainModels:
            # early_stopping
            early_stopping = EarlyStopping(patience=3, verbose=True)
            NNmodel = model[3].loading_NeuralNetwork(sys.argv[1])
            NNmodel.eval()
            print("Results model na preprocessing " + ' ' + model[4])
            calculatePrecisionDev(model[1], NNmodel, oneHotEncoderM, domainLabels, domainLabelIndices, device)
            validation_loss, microF1, macroF1 = eval_loop(model[1], NNmodel, oneHotEncoderM, domainLabels,
                                                          domainLabelIndices, device)
            early_stopping(validation_loss, microF1, macroF1, NNmodel)
            optimizer = torch.optim.Adam(NNmodel.parameters(), lr=1e-4)
            numberOfTrainingSteps = len(model[0])
            if numberOfTrainingSteps % 32 == 0:
                numberOfTrainingSteps = numberOfTrainingSteps / 32 * epochs * 100
            else:
                numberOfTrainingSteps = (numberOfTrainingSteps // 32 + 1) * epochs * 100
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0,
                num_training_steps=numberOfTrainingSteps
            )
            models.add(
                tuple([iter(model[0]), model[3], optimizer, model[4],
                       model[5], model[2], 0, model[0], early_stopping,
                       model[1], scheduler]))

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
                    if model[10] != "none":
                        model[10].step()
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
                             model[9], model[10]]))
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