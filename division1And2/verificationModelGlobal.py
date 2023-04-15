import os
import pickle
import sys
import random

import numpy as np
from transformers import get_linear_schedule_with_warmup

from division1And2.pytorchtools import EarlyStopping

import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

from division1And2.labelMaskDomain import labelMaskDomain

from dataset import dump_load, dump_write, NUS

from division1And2.encoderMetadata import encoderMetadata
from division1And2.evidence_ranker import evidenceRanker
from division1And2.instanceEncoder import instanceEncoder
from division1And2.labelEmbeddingLayer import labelEmbeddingLayer
from division1And2.OneHotEncoder import oneHotEncoder
from torch import nn
from sklearn.metrics import f1_score

from division1And2.encoderGlobal import encoder


class verifactionModel(nn.Module):
    # Create neural network
    def __init__(self,encoder,metadataEncoder,instanceEncoder,evidenceRanker,labelEmbedding,labelMaskDomain,labelDomains,domain,alpha,beta,withPretext=False):
        super(verifactionModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder
        self.metaDataEncoder = metadataEncoder
        self.instanceEncoder = instanceEncoder
        self.evidenceRanker = evidenceRanker
        self.labelEmbedding = labelEmbedding
        self.labelDomains = labelDomains
        self.labelMaskDomain = labelMaskDomain
        self.softmax = torch.nn.Softmax(dim=0).to(self.device)
        self.domain = domain
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.withPreText = withPretext

    def forward(self,claim,evidences,metadata_encoding,domain,claimDate,snippetDates,verbsClaim,timeExpressionsClaim,positionClaim,timeRefsClaim,
                timeHeidelClaim,verbsSnippets,timeExpressionsSnippets,positionSnippets,timeRefsSnippets,timeHeidelSnippets,sizePretextClaim, sizePretextSnippets):
        if self.withPreText:
            sizePretextClaim = 0
        claim_encoding = self.encoder(claim,claimDate,positionClaim.split('\t'),verbsClaim.split('\t'),timeExpressionsClaim.split('\t'),timeRefsClaim.split('\t'),timeHeidelClaim.split('\t'),sizePretextClaim).to(self.device)
        distribution = torch.zeros(len(self.labelDomains[domain])).to(self.device)
        evidences = evidences.split(' 0123456789 ')[:-1]
        verbsSnippets = verbsSnippets.split(' 0123456789 ')[:-1]
        timeExpressionsSnippets = timeExpressionsSnippets.split(' 0123456789 ')[:-1]
        positionSnippets = positionSnippets.split(' 0123456789 ')[:-1]
        timeRefsSnippets = timeRefsSnippets.split(' 0123456789 ')[:-1]
        timeHeidelSnippets = timeHeidelSnippets.split(' 0123456789 ')[:-1]
        sizePretextSnippet = sizePretextSnippets.split(' 0123456789 ')[:-1]
        snippetDates = snippetDates.split('\t')
        for i in range(len(evidences)):
            positionSnippet = positionSnippets[i].split('\t')
            verbsSnippet = verbsSnippets[i].split('\t')
            timeExpressionsSnippet = timeExpressionsSnippets[i].split('\t')
            timeRefsSnippet = timeRefsSnippets[i].split('\t')
            timeHeidelSnippet = timeHeidelSnippets[i].split('\t')
            if self.withPreText:
                sizePretextSnippet[i]=0
            evidence_encoding = self.encoder(evidences[i],int(snippetDates[i+1]),positionSnippet,verbsSnippet,timeExpressionsSnippet,
                                             timeRefsSnippet,timeHeidelSnippet,sizePretextSnippet[i],isClaim=False).to(self.device)
            instance_encoding = self.instanceEncoder(claim_encoding.squeeze(0),evidence_encoding.squeeze(0),metadata_encoding.squeeze(0)).to(self.device)
            rank_evidence = self.evidenceRanker(instance_encoding).to(self.device)
            label_distribution = self.labelEmbedding(instance_encoding,domain).to(self.device)
            label_distributionDomain = self.labelMaskDomain(label_distribution).to(self.device)
            distribution = distribution + torch.mul(rank_evidence,label_distributionDomain).to(self.device)
        return distribution.to(self.device)

    def forwardAttribution(self,claim,evidences,metadata_encoding,domain,claimDate,snippetDates,verbsClaim,timeExpressionsClaim,positionClaim,timeRefsClaim,
                timeHeidelClaim,verbsSnippets,timeExpressionsSnippets,positionSnippets,timeRefsSnippets,timeHeidelSnippets,sizePretextClaim, sizePretextSnippets):
        if self.withPreText:
            sizePretextClaim = 0
        claim_encodingFull, claim_EncodingWithoutTime, times,verschilTimeClaim = self.encoder.forwardAttribution(claim,claimDate,positionClaim.split('\t'),verbsClaim.split('\t'),timeExpressionsClaim.split('\t'),timeRefsClaim.split('\t'),timeHeidelClaim.split('\t'),sizePretextClaim)
        distribution = torch.zeros(len(self.labelDomains[self.domain])).to(self.device)
        evidences = evidences.split(' 0123456789 ')[:-1]
        verbsSnippets = verbsSnippets.split(' 0123456789 ')[:-1]
        timeExpressionsSnippets = timeExpressionsSnippets.split(' 0123456789 ')[:-1]
        positionSnippets = positionSnippets.split(' 0123456789 ')[:-1]
        timeRefsSnippets = timeRefsSnippets.split(' 0123456789 ')[:-1]
        timeHeidelSnippets = timeHeidelSnippets.split(' 0123456789 ')[:-1]
        sizePretextSnippet = sizePretextSnippets.split(' 0123456789 ')[:-1]
        snippetDates = snippetDates.split('\t')
        evidenceEncodings = []
        for i in range(len(evidences)):
            positionSnippet = positionSnippets[i].split('\t')
            verbsSnippet = verbsSnippets[i].split('\t')
            timeExpressionsSnippet = timeExpressionsSnippets[i].split('\t')
            timeRefsSnippet = timeRefsSnippets[i].split('\t')
            timeHeidelSnippet = timeHeidelSnippets[i].split('\t')
            if self.withPreText:
                sizePretextSnippet[i]=0
            evidence_encoding, evidence_EncodingWithoutTime, timeEvidences,verschiTimeEvidence = self.encoder.forwardAttribution(
                evidences[i], int(snippetDates[i + 1]), positionSnippet, verbsSnippet, timeExpressionsSnippet,
                timeRefsSnippet, timeHeidelSnippet,sizePretextSnippet[i], isClaim=False)
            evidenceEncodings.append((evidence_encoding, evidence_EncodingWithoutTime, timeEvidences,verschiTimeEvidence))
            instance_encoding = self.instanceEncoder(claim_encodingFull, evidence_encoding,
                                                     metadata_encoding.squeeze(0)).to(self.device)
            rank_evidence = self.evidenceRanker(instance_encoding).to(self.device)
            label_distribution = self.labelEmbedding(instance_encoding, self.domain).to(self.device)
            label_distributionDomain = self.labelMaskDomain(label_distribution).to(self.device)
            distribution = distribution + torch.mul(rank_evidence, label_distributionDomain).to(self.device)
        return distribution.to(self.device), claim_encodingFull, claim_EncodingWithoutTime, times, evidenceEncodings,verschilTimeClaim

    def forwardIntegrated(self, inputs, metadata_encoding):
        if len(inputs[1])>0:
            sum = torch.zeros(len(inputs[1][0]))
            number = 0
            for j in range(len(inputs[1])):
                sum += inputs[1][j]
                number += 1
            claimEncoding = self.alpha * inputs[0] + self.beta*inputs[2]+(1-self.alpha-self.beta)*0.75 * sum/number
        else:
            claimEncoding = self.alpha * inputs[0] + self.beta*inputs[2]
        distribution = torch.zeros(len(self.labelDomains[self.domain])).to(self.device)
        for i in range(len(inputs[3])):
            if len(inputs[3][i][1])>0:
                sum = torch.zeros(len(inputs[3][i][1][0]))
                number = 0
                for j in range(len(inputs[3][i][1])):
                    sum += inputs[3][i][1][j]
                    number += 1
                evidence_encoding = self.alpha* inputs[3][i][0]+self.beta*inputs[3][i][2] + 0.75 * sum/number
            else:
                evidence_encoding = self.alpha* inputs[3][i][0]+self.beta*inputs[3][i][2]
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
            print(distribution)
            differenceToZero = torch.max(distribution).item()
            print(differenceToZero)
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

    def getRankingLabelsPerBin(self, index, claim, evidences, metadata_encoding, domain, claimDate, snippetDates,
                               verbsClaim, timeExpressionsClaim, positionClaim, timeRefsClaim,
                               timeHeidelClaim, verbsSnippets, timeExpressionsSnippets, positionSnippets,
                               timeRefsSnippets, timeHeidelSnippets,sizePretextClaim, sizePretextSnippets):
        labelsAllTime = {}
        labelsDomainTime = {}
        labelsAllAbsolute = {}
        labelsAllAbsoluteIndices = {}
        labelsAbsoluteDomain = {}
        labelsAbsoluteDomainIndices = {}
        times = {}
        if self.withPreText:
            sizePretextClaim = 0
        claim_encoding = self.encoder(claim,claimDate,positionClaim.split('\t'),verbsClaim.split('\t'),
                                      timeExpressionsClaim.split('\t'),timeRefsClaim.split('\t'),timeHeidelClaim.split('\t'),sizePretextClaim).to(self.device)
        evidences = evidences.split(' 0123456789 ')[:-1]
        verbsSnippets = verbsSnippets.split(' 0123456789 ')[:-1]
        timeExpressionsSnippets = timeExpressionsSnippets.split(' 0123456789 ')[:-1]
        positionSnippets = positionSnippets.split(' 0123456789 ')[:-1]
        timeRefsSnippets = timeRefsSnippets.split(' 0123456789 ')[:-1]
        timeHeidelSnippets = timeHeidelSnippets.split(' 0123456789 ')[:-1]
        sizePretextSnippet = sizePretextSnippets.split(' 0123456789 ')[:-1]
        snippetDates = snippetDates.split('\t')
        for i in range(len(evidences)):
            positionSnippet = positionSnippets[i].split('\t')
            verbsSnippet = verbsSnippets[i].split('\t')
            timeExpressionsSnippet = timeExpressionsSnippets[i].split('\t')
            timeRefsSnippet = timeRefsSnippets[i].split('\t')
            timeHeidelSnippet = timeHeidelSnippets[i].split('\t')
            if self.withPreText:
                sizePretextSnippet[i]=0
            evidence_encoding = self.encoder(evidences[i], int(snippetDates[i + 1]), positionSnippet, verbsSnippet,
                                             timeExpressionsSnippet,
                                             timeRefsSnippet, timeHeidelSnippet,sizePretextSnippet[i], isClaim=False).to(self.device)
            instance_encoding = self.instanceEncoder(claim_encoding, evidence_encoding,
                                                     metadata_encoding.squeeze(0)).to(self.device)
            timeSnippetsEntry = set()

            for x in range(len(timeHeidelSnippet)):
                if timeHeidelSnippet[x].find('Duur') == -1 and timeHeidelSnippet[x].find('Refs') == -1:
                    if timeHeidelSnippet[x].isdigit():
                        timeSnippetsEntry.add(timeHeidelSnippet[x])
            timeSnippetsEntry = list(timeSnippetsEntry)
            for x in range(len(timeSnippetsEntry)):
                for j in range(x+1,len(timeSnippetsEntry)):
                    if (timeSnippetsEntry[x],timeSnippetsEntry[j]) in times:
                        times[(timeSnippetsEntry[x],timeSnippetsEntry[j])] += 1
                    else:
                        times[(timeSnippetsEntry[x], timeSnippetsEntry[j])] = 1
            rank_evidence = self.evidenceRanker(instance_encoding).to(self.device)
            label_distribution = self.labelEmbedding(instance_encoding, domain).to(self.device)
            ranking = label_distribution.cpu().detach().tolist()
            labelsRanking = [sorted(ranking, reverse=True).index(x) + 1 for x in ranking]
            if int(snippetDates[i + 1]) in labelsAllTime:
                labelsAllTime[int(snippetDates[i + 1])].append(labelsRanking)
            else:
                labelsAllTime[int(snippetDates[i + 1])] = [labelsRanking]
            for date in timeSnippetsEntry:
                if date in labelsAllAbsolute:
                    labelsAllAbsolute[date].append(labelsRanking)
                    labelsAllAbsoluteIndices[date].append(index + "-" + str(i))
                else:
                    labelsAllAbsolute[date] = [labelsRanking]
                    labelsAllAbsoluteIndices[date] = [index + "-" + str(i)]
            label_distributionDomain = self.labelMaskDomain(label_distribution).to(self.device)
            rankingDomain = label_distributionDomain.cpu().detach().tolist()
            labelsRankingDomain = [sorted(rankingDomain, reverse=True).index(x) + 1 for x in rankingDomain]
            for date in timeSnippetsEntry:
                if date in labelsAbsoluteDomain:
                    labelsAbsoluteDomain[date].append(labelsRankingDomain)
                    labelsAbsoluteDomainIndices[date].append(index + "-" + str(i))
                else:
                    labelsAbsoluteDomain[date] = [labelsRankingDomain]
                    labelsAbsoluteDomainIndices[date] = [index + "-" + str(i)]
            if int(snippetDates[i + 1]) in labelsDomainTime:
                labelsDomainTime[int(snippetDates[i + 1])].append(labelsRankingDomain)
            else:
                labelsDomainTime[int(snippetDates[i + 1])] = [labelsRankingDomain]
        return labelsAbsoluteDomain, labelsAllAbsolute, labelsAbsoluteDomainIndices, labelsAllAbsoluteIndices,labelsAllTime,labelsDomainTime,times

    def getRankingEvidencesLabels(self,claim,evidences,metadata_encoding,domain,claimDate,snippetDates,verbsClaim,timeExpressionsClaim,positionClaim,timeRefsClaim,
                timeHeidelClaim,verbsSnippets,timeExpressionsSnippets,positionSnippets,timeRefsSnippets,timeHeidelSnippets,sizePretextClaim, sizePretextSnippets):
        if self.withPreText:
            sizePretextClaim = 0
        claim_encoding = self.encoder(claim, claimDate, positionClaim.split('\t'), verbsClaim.split('\t'),
                                      timeExpressionsClaim.split('\t'), timeRefsClaim.split('\t'),
                                      timeHeidelClaim.split('\t'),sizePretextClaim).to(self.device)
        ranking = []
        labelsAll = []
        labelsDomain = []
        distribution = torch.zeros(len(self.labelDomains[domain])).to(self.device)
        evidences = evidences.split(' 0123456789 ')[:-1]
        verbsSnippets = verbsSnippets.split(' 0123456789 ')[:-1]
        timeExpressionsSnippets = timeExpressionsSnippets.split(' 0123456789 ')[:-1]
        positionSnippets = positionSnippets.split(' 0123456789 ')[:-1]
        timeRefsSnippets = timeRefsSnippets.split(' 0123456789 ')[:-1]
        timeHeidelSnippets = timeHeidelSnippets.split(' 0123456789 ')[:-1]
        sizePretextSnippet = sizePretextSnippets.split(' 0123456789 ')[:-1]
        snippetDates = snippetDates.split('\t')
        lastInstance_encoding = "None"
        allEqual = True
        for i in range(len(evidences)):
            positionSnippet = positionSnippets[i].split('\t')
            verbsSnippet = verbsSnippets[i].split('\t')
            timeExpressionsSnippet = timeExpressionsSnippets[i].split('\t')
            timeRefsSnippet = timeRefsSnippets[i].split('\t')
            timeHeidelSnippet = timeHeidelSnippets[i].split('\t')
            timeHeidelSnippet = timeHeidelSnippets[i].split('\t')
            if self.withPreText:
                sizePretextSnippet[i] = 0
            evidence_encoding = self.encoder(evidences[i], int(snippetDates[i + 1]), positionSnippet, verbsSnippet,
                                             timeExpressionsSnippet,
                                             timeRefsSnippet, timeHeidelSnippet,sizePretextSnippet[i], isClaim=False).to(self.device)
            instance_encoding = self.instanceEncoder(claim_encoding.squeeze(0), evidence_encoding.squeeze(0),
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

    def getClaimEncoding(self, claim):
        return self.encoder.getEncodingWithoutTime(claim).to(self.device)

    def getSnippetEncodings(self, evidences, snippetDates):
        evidences = evidences.split(' 0123456789 ')[:-1]
        evidenceEncodings = torch.tensor([])
        for i in range(len(evidences)):
            evidence_encoding = self.encodergetEncodingWithoutTime(evidences[i]).to(self.device)
            evidenceEncodings = torch.cat((evidenceEncodings, evidence_encoding))
        return evidenceEncodings

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
    loss = nn.CrossEntropyLoss(reduction="sum")
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
                prediction = model(c[1][i], c[2][i], metadata_encoding, domain, c[5][i], c[6][i],
                                   c[7][i],
                                   c[8][i], c[9][i], c[10][i], c[11][i], c[12][i], c[13][i],
                                   c[14][i], c[15][i], c[16][i],c[17][i],c[18][i]).to(device)
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

def getLabelIndicesDomain(domainPath,labelPath):
    domainsIndices = dict()
    domainsLabels = dict()
    domainLabelIndices = dict()
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

    return domainsIndices,domainsLabels,domainLabelIndices

def calculatePrecisionDev(dataloader, model,oneHotEncoder,domainLabels,domainLabelIndices,device):
    groundTruthLabels = []
    predictedLabels = []
    with torch.no_grad():
        for batch in dataloader:
            for i in range(len(batch[0])):
                metaDataClaim = oneHotEncoder.encode(batch[3][i],device)
                metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                domain = batch[0][0].split('-')[0]
                prediction = model(batch[1][i], batch[2][i], metadata_encoding, domain, batch[5][i], batch[6][i],
                                   batch[7][i],
                                   batch[8][i], batch[9][i], batch[10][i], batch[11][i], batch[12][i], batch[13][i],
                                   batch[14][i], batch[15][i], batch[16][i],batch[17][i],batch[18][i]).to(device)
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
                                   batch[14][i], batch[15][i], batch[16][i],batch[17][i],batch[18][i]).to(device)
                pIndex = torch.argmax(prediction).item()
                plabel = domainLabels[domain][domainLabelIndices[domain].index(pIndex)]
                predictions[batch[0][i]] = plabel
    return predictions

def trainFinetuning(batch,model,oneHotEncoder, optimizer,domainLabels,domainLabelIndices,device,preprocessing = False):
    model.train()
    loss = nn.CrossEntropyLoss(reduction="sum")
    predictionBatch = torch.tensor([],requires_grad=True).to(device)
    for i in range(len(batch[0])):
        metaDataClaim = oneHotEncoder.encode(batch[3][i],device).to(device)
        metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
        domain = batch[0][0].split('-')[0]
        prediction = model(batch[1][i], batch[2][i], metadata_encoding, domain,batch[5][i],batch[6][i],batch[7][i],
                           batch[8][i],batch[9][i],batch[10][i],batch[11][i],batch[12][i],batch[13][i],
                           batch[14][i],batch[15][i],batch[16][i],batch[17][i],batch[18][i]).to(device)
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
    torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
    #  adjust the parameters by the gradients collected in the backward pass
    optimizer.step()
    return output.item()

def train(batch,model,oneHotEncoder, optimizer,domainLabels,domainLabelIndices,device,preprocessing = False):
    model.train()
    loss = nn.CrossEntropyLoss(reduction="sum")
    predictionBatch = torch.tensor([],requires_grad=True).to(device)
    for i in range(len(batch[0])):
        metaDataClaim = oneHotEncoder.encode(batch[3][i],device).to(device)
        metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
        domain = batch[0][0].split('-')[0]
        prediction = model(batch[1][i], batch[2][i], metadata_encoding, domain,batch[5][i],batch[6][i],batch[7][i],
                           batch[8][i],batch[9][i],batch[10][i],batch[11][i],batch[12][i],batch[13][i],
                           batch[14][i],batch[15][i],batch[16][i],batch[17][i],batch[18][i]).to(device)
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

def preprocessing(models,epochs=600):
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
                 2 parameter alpha
                 3 parameter beta
                 4 evaluation/training mode
                 5 with pretext
    '''
    torch.manual_seed(1)
    random.seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    domainIndices, domainLabels, domainLabelIndices = getLabelIndicesDomain(
        'labels/labels.tsv', 'labels/labelSequence','labels/weights.tsv')
    oneHotEncoderM = oneHotEncoder('Metadata_sequence/metadata')
    domains = domainIndices.keys()
    metadataSet = set()
    labelEmbeddingLayerM = labelEmbeddingLayer(772, domainIndices)
    domainModels = []
    losses = np.zeros(len(domains))
    index = 0
    encoderM = encoder(300, 128,sys.argv[2],sys.argv[3]).to(device)
    encoderMetadataM = encoderMetadata(3, 3, oneHotEncoderM).to(device)
    instanceEncoderM = instanceEncoder().to(device)
    evidenceRankerM = evidenceRanker(772, 100).to(device)

    for domain in domains:
        train_set = NUS(mode='Train', path='train/train-' + domain + '.tsv', domain=domain)
        dev_set = NUS(mode='Dev', path='dev/dev-' + domain + '.tsv', domain=domain)
        test_set = NUS(mode='Test', path='test/test-' + domain + '.tsv', domain=domain)

        trainMetadata = train_set.getMetaDataSet()
        devMetadata = dev_set.getMetaDataSet()
        testMetadata = test_set.getMetaDataSet()
        metadataSet = set.union(metadataSet,trainMetadata)
        metadataSet = set.union(metadataSet,devMetadata)
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
        labelMaskDomainM = labelMaskDomain(772,domainIndices,domain,len(domainIndices[domain])).to(device)

        verificationModelM = verifactionModel(encoderM, encoderMetadataM, instanceEncoderM,
                                            evidenceRankerM,
                                            labelEmbeddingLayerM,labelMaskDomainM, domainIndices,domain,
                                              sys.argv[2],sys.argv[3],bool(sys.argv[5])).to(device)
        domainModel = [train_loader, dev_loader, test_loader, verificationModelM, domain, index]
        domainModels.append(domainModel)
        index += 1

    if sys.argv[4] == "evaluation":
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
                validation_loss, microF1, macroF1 = eval_loop(model[1], NNmodel, oneHotEncoderM, domainLabels,
                                                              domainLabelIndices, device)
                early_stopping(validation_loss, microF1, macroF1, NNmodel)
                optimizer = torch.optim.RMSprop(NNmodel.parameters(), lr=2e-4)
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
                optimizer = torch.optim.RMSprop(NNmodel.parameters(), lr=2e-4)
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
                        loss = train(batch, model[1], oneHotEncoderM, model[2], domainLabels, domainLabelIndices,
                                     device)
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
