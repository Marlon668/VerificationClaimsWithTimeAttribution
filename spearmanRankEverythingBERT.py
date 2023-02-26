import math
import os
import sys
import random
import pickle
import scipy.stats as ss
import numpy
import numpy as np
import torch.nn.functional as F
from scipy import stats
from dataset import NUS
from division1And2.verificationModelBERTGlobal import verifactionModel as verificationEverything
from division1And2 import OneHotEncoder, labelEmbeddingLayer, encoderMetadata, \
    instanceEncoder, evidence_ranker, labelMaskDomain
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoModel, AutoTokenizer

'''
    Calculate intra and inter SpearmanRankingCoefficient for division1And2 according to experiment 3
    for DistilRoBERTa as encoder
'''
def spearmanRanking(loaders,models):
    labelsTimeBins = [{}, {}]
    labelsTimeBinsDomain = [{}, {}]
    labelstimeTextBins = [{},{}]
    labelstimeTextBinsDomain = [{},{}]
    labelstimeTextBinsIndices = [{}, {}]
    labelstimeTextBinsDomainIndices = [{}, {}]
    timestimeText = {}
    for loader in loaders:
        for data in loader[0]:
            for i in range(len(data[0])):
                for j in range(0,len(models)):
                    model = models[j]
                    metaDataClaim = model.metaDataEncoder.oneHotEncoder.encode(data[3][i], device)
                    metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                    labelsDomaintimeText,labelsAlltimeText,labelsDomainIndices,labelsAllIndices,lablsAllTime,labelsDomainTime,timeTextTimeOccurence = model.getRankingLabelsPerBin(data[0][i],data[1][i], data[2][i],
                                                                                                     metadata_encoding, domain,
                                                                                                     data[5][i], data[6][i],
                                                                                                     data[7][i],
                          data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                          data[14][i], data[15][i], data[16][i])
                    if j == 0:
                        for k, v in timeTextTimeOccurence.items():
                            if (k[0],k[1]) in timestimeText.keys():
                                timestimeText[(k[0],k[1])] += v
                            else:
                                if (k[1],k[0]) in timestimeText.keys():
                                    timestimeText[(k[1],k[0])] += v
                                else:
                                    timestimeText[k] = v

                    for k, v in labelsAlltimeText.items():
                        if k in labelstimeTextBins[j].keys():
                            labelstimeTextBins[j][k] += v
                        else:
                            labelstimeTextBins[j][k] = v
                    for k, v in labelsAllIndices.items():
                        if k in labelstimeTextBinsIndices[j].keys():
                            labelstimeTextBinsIndices[j][k] += v
                        else:
                            labelstimeTextBinsIndices[j][k] = v
                    for k, v in labelsDomaintimeText.items():
                        if k in labelstimeTextBinsDomain[j].keys():
                            labelstimeTextBinsDomain[j][k] += v
                        else:
                            labelstimeTextBinsDomain[j][k] = v
                    for k, v in labelsDomainIndices.items():
                        if k in labelstimeTextBinsDomainIndices[j].keys():
                            labelstimeTextBinsDomainIndices[j][k] += v
                        else:
                            labelstimeTextBinsDomainIndices[j][k] = v
                    for k, v in lablsAllTime.items():
                        if k in labelsTimeBins[j].keys():
                            labelsTimeBins[j][k] += v
                        else:
                            labelsTimeBins[j][k] = v
                    for k, v in labelsDomainTime.items():
                        if k in labelsTimeBinsDomain[j].keys():
                            labelsTimeBinsDomain[j][k] += v
                        else:
                            labelsTimeBinsDomain[j][k] = v
    return labelsTimeBins,labelsTimeBinsDomain,labelstimeTextBins,labelstimeTextBinsDomain,labelstimeTextBinsIndices,labelstimeTextBinsDomainIndices,timestimeText

def getIntraRankingLabelsAll(labelsAll):
    spearmanLabelsAll = [{},{}]
    for model in range(0,len(spearmanLabelsAll)):
        for time in labelsAll[model]:
            if(len(labelsAll[model][time]))>1:
                correlation,_ = stats.spearmanr(labelsAll[model][time],axis=1)
                if len(labelsAll[model][time]) == 2:
                    if time in spearmanLabelsAll[model]:
                        spearmanLabelsAll[model][time].append(correlation)
                    else:
                        spearmanLabelsAll[model][time] = [correlation]
                else:
                    for i in range(0,len(correlation)):
                        for j in range(i+1,len(correlation[i])):
                            if time in spearmanLabelsAll[model]:
                                spearmanLabelsAll[model][time].append(correlation[i][j])
                            else:
                                spearmanLabelsAll[model][time] = [correlation[i][j]]
    return spearmanLabelsAll

def getInterRankingLabelsDomain(labelsDomain,domains):
    spearmanLabelsDomainAll = [{},{}]
    spearmanLabelsDomain = [{},{}]
    for domain in domains:
        times = list(labelsDomain[domain][0])
        for model in range(0, len(labelsDomain[domain])):
            spearmanLabelsDomain[model][domain] = {}
            for x in range(0, len(times)):
                time1 = times[x]
                for y in range(x + 1, len(times)):
                    time2 = times[y]
                    correlation,_ = stats.spearmanr(labelsDomain[domain][model][time1], labelsDomain[domain][model][time2],axis=1)
                    if(len(labelsDomain[domain][model][time1])+len(labelsDomain[domain][model][time2]))==2:
                        if (time1, time2) in spearmanLabelsDomain[model][domain]:
                            spearmanLabelsDomain[model][domain][(time1, time2)].append(correlation)
                        else:
                            spearmanLabelsDomain[model][domain][(time1, time2)] = [correlation]
                        if (time1, time2) in spearmanLabelsDomainAll[model]:
                            spearmanLabelsDomainAll[model][(time1, time2)].append(
                                correlation)
                        else:
                            spearmanLabelsDomainAll[model][(time1, time2)] = [
                                correlation]
                    else:
                        for i in range(0,len(labelsDomain[domain][model][time1])):
                            for j in range(len(labelsDomain[domain][model][time1]),len(correlation[0])):
                                if (time1,time2) in spearmanLabelsDomain[model][domain]:
                                    spearmanLabelsDomain[model][domain][(time1,time2)].append(correlation[i][j])
                                else:
                                    spearmanLabelsDomain[model][domain][(time1,time2)] = [correlation[i][j]]
                                if (time1,time2) in spearmanLabelsDomainAll[model]:
                                    spearmanLabelsDomainAll[model][(time1,time2)].append(
                                        correlation[i][j])
                                else:
                                    spearmanLabelsDomainAll[model][(time1,time2)] = [
                                        correlation[i][j]]
    return spearmanLabelsDomainAll,spearmanLabelsDomain

def getIntraRankingLabelsDomain(labelsDomain,domains):
    spearmanLabelsDomainAll = [{},{}]
    spearmanLabelsDomain = [{},{}]
    for domain in domains:
        for model in range(0, len(labelsDomain[domain])):
            spearmanLabelsDomain[model][domain] = {}
            for time in labelsDomain[domain][model]:
                if (len(labelsDomain[domain][model][time])) > 1:
                    correlation, _ = stats.spearmanr(labelsDomain[domain][model][time], axis=1)
                    if (len(labelsDomain[domain][model][time])) == 2:
                        if time in spearmanLabelsDomain[model][domain]:
                            spearmanLabelsDomain[model][domain][time].append(correlation)
                        else:
                            spearmanLabelsDomain[model][domain][time] = [correlation]
                        if time in spearmanLabelsDomainAll[model]:
                            spearmanLabelsDomainAll[model][time].append(
                                correlation)
                        else:
                            spearmanLabelsDomainAll[model][time] = [
                                correlation]
                    else:
                        for i in range(0,len(correlation)):
                            for j in range(i+1,len(correlation)):
                                if time in spearmanLabelsDomain[model][domain]:
                                    spearmanLabelsDomain[model][domain][time].append(correlation[i][j])
                                else:
                                    spearmanLabelsDomain[model][domain][time] = [correlation[i][j]]
                                if time in spearmanLabelsDomainAll[model]:
                                    spearmanLabelsDomainAll[model][time].append(
                                        correlation[i][j])
                                else:
                                    spearmanLabelsDomainAll[model][time] = [
                                        correlation[i][j]]
    return spearmanLabelsDomainAll,spearmanLabelsDomain

def getInterRankingLabelsAll(labelsAll):
    spearmanLabelsAll = [{},{}]
    times = list(labelsAll[0])
    for model in range(0,len(spearmanLabelsAll)):
        for x in range(0,len(times)):
            time1  = times[x]
            for y in range(x+1,len(times)):
                time2 = times[y]
                correlation, _ = stats.spearmanr(labelsAll[model][time1], labelsAll[model][time2],
                                                 axis=1)
                if (len(labelsAll[model][time1]) + len(labelsAll[model][time2])) == 2:
                    if (time1, time2) in spearmanLabelsAll[model]:
                        spearmanLabelsAll[model][(time1, time2)].append(correlation)
                    else:
                        spearmanLabelsAll[model][(time1, time2)] = [correlation]
                else:
                    for i in range(0, len(labelsAll[model][time1])):
                        for j in range(len(labelsAll[model][time1]), len(correlation[0])):
                            if (time1, time2) in spearmanLabelsAll[model]:
                                spearmanLabelsAll[model][(time1, time2)].append(correlation[i][j])
                            else:
                                spearmanLabelsAll[model][(time1, time2)] = [correlation[i][j]]
    return spearmanLabelsAll

def calculateMeanAndStdAll(spearmanLabelsAll):
    means = [{},{}]
    stds = [{},{}]
    for model in range(0,len(means)):
        for time in spearmanLabelsAll[model]:
            means[model][time] = np.mean(spearmanLabelsAll[model][time])
            stds[model][time] = np.std(spearmanLabelsAll[model][time])
    meansTogether = {}
    stdsTogether = {}
    for time in spearmanLabelsAll[0]:
        meansTogether[time] = (np.mean([means[0][time],means[1][time]]),np.std([means[0][time],means[1][time]]))
        stdsTogether[time] = (np.mean([stds[0][time],stds[1][time]]),np.std([stds[0][time],stds[1][time]]))
    return meansTogether,stdsTogether

def calculateMeanAndStdDomain(spearmanLabelsAll,domains):
    means = [{},{}]
    stds = [{},{}]
    for domain in domains:
        for model in range(0,len(means)):
            means[model][domain] = {}
            stds[model][domain] = {}
            for time in spearmanLabelsAll[model][domain]:
                means[model][domain][time] = np.mean(spearmanLabelsAll[model][domain][time])
                stds[model][domain][time] = np.std(spearmanLabelsAll[model][domain][time])
    meansTogether = {}
    stdsTogether = {}
    for domain in domains:
        meansTogether[domain] = {}
        stdsTogether[domain] = {}
        for time in spearmanLabelsAll[0][domain]:
            meansTogether[domain][time] = (np.mean([means[0][domain][time],means[1][domain][time]]),np.std([means[0][domain][time],means[1][domain][time]]))
            stdsTogether[domain][time] = (np.mean([stds[0][domain][time],stds[1][domain][time]]),np.std([stds[0][domain][time],stds[1][domain][time]]))
    return meansTogether,stdsTogether

def getIntraRankingLabelsAlltimeText(labelsAll):
    spearmanLabelsAll = [{},{}]
    for model in range(0,len(spearmanLabelsAll)):
        for time in labelsAll[model]:
            if (len(labelsAll[model][time])) > 1:
                correlation, _ = stats.spearmanr(labelsAll[model][time], axis=1)
                if len(labelsAll[model][time]) == 2:
                    if time in spearmanLabelsAll[model]:
                        spearmanLabelsAll[model][time].append(correlation)
                    else:
                        spearmanLabelsAll[model][time] = [correlation]
                else:
                    for i in range(0, len(correlation)):
                        for j in range(i + 1, len(correlation[i])):
                            if time in spearmanLabelsAll[model]:
                                spearmanLabelsAll[model][time].append(correlation[i][j])
                            else:
                                spearmanLabelsAll[model][time] = [correlation[i][j]]
    return spearmanLabelsAll

def getInterRankingLabelsDomaintimeText(labelsDomain,domains,indicesLabelsDomain):
    spearmanLabelsDomainAll = [{},{}]
    spearmanLabelsDomain = [{},{}]
    for domain in domains:
        times = list(labelsDomain[domain][0])
        for model in range(0, len(labelsDomain[domain])):
            spearmanLabelsDomain[model][domain] = {}
            for x in range(0, len(times)):
                time1 = times[x]
                for y in range(x + 1, len(times)):
                    time2 = times[y]
                    correlation, _ = stats.spearmanr(labelsDomain[domain][model][time1],
                                                     labelsDomain[domain][model][time2], axis=1)
                    if (len(labelsDomain[domain][model][time1]) + len(labelsDomain[domain][model][time2])) == 2:
                        if (time1, time2) in spearmanLabelsDomain[model][domain]:
                            spearmanLabelsDomain[model][domain][(time1, time2)].append(correlation)
                        else:
                            spearmanLabelsDomain[model][domain][(time1, time2)] = [correlation]
                        if (time1, time2) in spearmanLabelsDomainAll[model]:
                            spearmanLabelsDomainAll[model][(time1, time2)].append(
                                correlation)
                        else:
                            spearmanLabelsDomainAll[model][(time1, time2)] = [
                                correlation]
                    else:
                        for i in range(0, len(labelsDomain[domain][model][time1])):
                            for j in range(len(labelsDomain[domain][model][time1]), len(correlation[0])):
                                if indicesLabelsDomain[domain][model][time1][i] !=indicesLabelsDomain[domain][model][time2][j-len(labelsDomain[domain][model][time1])]:
                                    if (time1, time2) in spearmanLabelsDomain[model][domain]:
                                        spearmanLabelsDomain[model][domain][(time1, time2)].append(correlation[i][j])
                                    else:
                                        spearmanLabelsDomain[model][domain][(time1, time2)] = [correlation[i][j]]
                                    if (time1, time2) in spearmanLabelsDomainAll[model]:
                                        spearmanLabelsDomainAll[model][(time1, time2)].append(
                                            correlation[i][j])
                                    else:
                                        spearmanLabelsDomainAll[model][(time1, time2)] = [
                                            correlation[i][j]]
    return spearmanLabelsDomainAll,spearmanLabelsDomain

def getIntraRankingLabelsDomaintimeText(labelsDomain,domains):
    spearmanLabelsDomainAll = [{},{}]
    spearmanLabelsDomain = [{},{}]
    for domain in domains:
        for model in range(0, len(labelsDomain[domain])):
            spearmanLabelsDomain[model][domain] = {}
            for time in labelsDomain[domain][model]:
                if (len(labelsDomain[domain][model][time])) > 1:
                    correlation, _ = stats.spearmanr(labelsDomain[domain][model][time], axis=1)
                    if (len(labelsDomain[domain][model][time])) == 2:
                        if time in spearmanLabelsDomain[model][domain]:
                            spearmanLabelsDomain[model][domain][time].append(correlation)
                        else:
                            spearmanLabelsDomain[model][domain][time] = [correlation]
                        if time in spearmanLabelsDomainAll[model]:
                            spearmanLabelsDomainAll[model][time].append(
                                correlation)
                        else:
                            spearmanLabelsDomainAll[model][time] = [
                                correlation]
                    else:
                        for i in range(0, len(correlation)):
                            for j in range(i + 1, len(correlation)):
                                if time in spearmanLabelsDomain[model][domain]:
                                    spearmanLabelsDomain[model][domain][time].append(correlation[i][j])
                                else:
                                    spearmanLabelsDomain[model][domain][time] = [correlation[i][j]]
                                if time in spearmanLabelsDomainAll[model]:
                                    spearmanLabelsDomainAll[model][time].append(
                                        correlation[i][j])
                                else:
                                    spearmanLabelsDomainAll[model][time] = [
                                        correlation[i][j]]
    return spearmanLabelsDomainAll,spearmanLabelsDomain

def getInterRankingLabelsAlltimeText(labelsAll,indicesLabelsAll):
    spearmanLabelsAll = [{},{}]
    times = list(labelsAll[0])
    for model in range(0,len(spearmanLabelsAll)):
        for x in range(0,len(times)):
            time1  = times[x]
            for y in range(x+1,len(times)):
                time2 = times[y]
                correlation, _ = stats.spearmanr(labelsAll[model][time1], labelsAll[model][time2],
                                                 axis=1)
                if (len(labelsAll[model][time1]) + len(labelsAll[model][time2])) == 2:
                    if (time1, time2) in spearmanLabelsAll[model]:
                        spearmanLabelsAll[model][(time1, time2)].append(correlation)
                    else:
                        spearmanLabelsAll[model][(time1, time2)] = [correlation]
                else:
                    for i in range(0, len(labelsAll[model][time1])):
                        for j in range(len(labelsAll[model][time1]), len(correlation[0])):
                            if indicesLabelsAll[model][time1][i] != indicesLabelsAll[model][time2][j-len(labelsAll[model][time1])]:
                                if (time1, time2) in spearmanLabelsAll[model]:
                                    spearmanLabelsAll[model][(time1, time2)].append(correlation[i][j])
                                else:
                                    spearmanLabelsAll[model][(time1, time2)] = [correlation[i][j]]
    return spearmanLabelsAll

def calculateMeanAndStdAlltimeText(spearmanLabelsAll):
    means = [{},{}]
    stds = [{},{}]
    for model in range(0,len(means)):
        for time in spearmanLabelsAll[model]:
            means[model][time] = np.mean(spearmanLabelsAll[model][time])
            stds[model][time] = np.std(spearmanLabelsAll[model][time])
    meansTogether = {}
    stdsTogether = {}
    for time in spearmanLabelsAll[0]:
        meansTogether[time] = (np.mean([means[0][time],means[1][time]]),np.std([means[0][time],means[1][time]]))
        stdsTogether[time] = (np.mean([stds[0][time],stds[1][time]]),np.std([stds[0][time],stds[1][time]]))
    return meansTogether,stdsTogether

def calculateMeanAndStdDomaintimeText(spearmanLabelsAll,domains):
    means = [{},{}]
    stds = [{},{}]
    for domain in domains:
        for model in range(0,len(means)):
            means[model][domain] = {}
            stds[model][domain] = {}
            for time in spearmanLabelsAll[model][domain]:
                means[model][domain][time] = np.mean(spearmanLabelsAll[model][domain][time])
                stds[model][domain][time] = np.std(spearmanLabelsAll[model][domain][time])
    meansTogether = {}
    stdsTogether = {}
    for domain in domains:
        meansTogether[domain] = {}
        stdsTogether[domain] = {}
        for time in spearmanLabelsAll[0][domain]:
            meansTogether[domain][time] = (np.mean([means[0][domain][time],means[1][domain][time]]),np.std([means[0][domain][time],means[1][domain][time]]))
            stdsTogether[domain][time] = (np.mean([stds[0][domain][time],stds[1][domain][time]]),np.std([stds[0][domain][time],stds[1][domain][time]]))
    return meansTogether,stdsTogether

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

'''
argument 1 path of first model division1And2
argument 2 path of second model division1And2
argument 3 parameter alpha
argument 4 parameter beta
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
domainIndices, domainLabels, domainLabelIndices, domainWeights = getLabelIndicesDomain(
    'labels/labels.tsv', 'labels/labelSequence', 'labels/weights.tsv')
domains = domainIndices.keys()
models = []
for domain in domains:
    test_set = NUS(mode='Test', path='test/test-' + domain + '.tsv',
                   pathToSave="test/time/dataset2/",
                   domain=domain)
    test_loader = DataLoader(test_set,
                            batch_size=1,
                            shuffle=False)
    models.append([test_loader,domain])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
labelsAllAllTime = [{},{}]
labelsBinsDomainTime = {}
labelsDomainAlltimeText = {}
labelsAllAlltimeText = [{},{}]
labelsDomainAllIndicestimeText = {}
labelsAllAllIndicestimeText = [{},{}]
timesDomainOccurenceA = {}
timesAllOccurenceA = {}

with torch.no_grad():
    for model in models:
        oneHotEncoderM = OneHotEncoder.oneHotEncoder('timeModels/Metadata_sequence/metadata')
        labelEmbeddingLayerEverything = labelEmbeddingLayer.labelEmbeddingLayer(2308, domainIndices)
        encoderMetadataEverything = encoderMetadata.encoderMetadata(3, 3, oneHotEncoderM).to(device)
        instanceEncoderEverything = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerEverything = evidence_ranker.evidenceRanker(2308, 100).to(device)
        labelMaskDomainEverything = labelMaskDomain.labelMaskDomain(2308, domainIndices, model[1],
                                                                         len(domainIndices[model[1]])).to(device)
        transformer = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
        verificationModelTimeEverything2020A = verificationEverything(transformer, encoderMetadataEverything,
                                                                      instanceEncoderEverything,
                                                                      evidenceRankerEverything,
                                                                      labelEmbeddingLayerEverything,
                                                                      labelMaskDomainEverything,
                                                                      domainIndices, domainWeights,
                                                                      domain, sys.argv[3], sys.argv[4])
        verificationModelTimeEverything2020A.loading_NeuralNetwork(sys.argv[1]).to(device)
        oneHotEncoderM = OneHotEncoder.oneHotEncoder('timeModels/Metadata_sequence/metadata')
        labelEmbeddingLayerEverything = labelEmbeddingLayer.labelEmbeddingLayer(2308, domainIndices)
        encoderMetadataEverything = encoderMetadata.encoderMetadata(3, 3, oneHotEncoderM).to(device)
        instanceEncoderEverything = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerEverything = evidence_ranker.evidenceRanker(2308, 100).to(device)
        labelMaskDomainEverything = labelMaskDomain.labelMaskDomain(2308, domainIndices, model[1],
                                                                    len(domainIndices[model[1]])).to(device)
        transformer = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
        verificationModelTimeEverything2020B = verificationEverything(transformer, encoderMetadataEverything,
                                                                      instanceEncoderEverything,
                                                                      evidenceRankerEverything,
                                                                      labelEmbeddingLayerEverything,
                                                                      labelMaskDomainEverything,
                                                                      domainIndices, domainWeights,
                                                                      domain, sys.argv[3], sys.argv[4])
        verificationModelTimeEverything2020B.loading_NeuralNetwork(sys.argv[2]).to(device)

        timeModels = [verificationModelTimeEverything2020A,verificationModelTimeEverything2020B]
        labelsTimeBins,labelsTimeBinsDomain,labelstimeTextBins,labelstimeTextBinsDomain,labelstimeTextBinsIndices,labelstimeTextBinsDomainIndices,timestimeTextOccurence = spearmanRanking([model],timeModels)
        timesDomainOccurenceA[model[1]] = timestimeTextOccurence
        labelsBinsDomainTime[model[1]] = labelsTimeBinsDomain
        labelsDomainAlltimeText[model[1]] = labelstimeTextBinsDomain
        labelsDomainAllIndicestimeText[model[1]] = labelstimeTextBinsDomainIndices
        for k, v in timestimeTextOccurence.items():
            if (k[0], k[1]) in timesAllOccurenceA.keys():
                timesAllOccurenceA[(k[0], k[1])] += v
            else:
                if (k[1], k[0]) in timesAllOccurenceA.keys():
                    timesAllOccurenceA[(k[1], k[0])] += v
                else:
                    timesAllOccurenceA[k] = v
        for j in range(0,2):
            for k, v in labelstimeTextBins[j].items():
                if k in labelsAllAlltimeText[j].keys():
                    labelsAllAlltimeText[j][k] += v
                else:
                    labelsAllAlltimeText[j][k] = v
        for j in range(0,2):
            for k, v in labelstimeTextBinsIndices[j].items():
                if k in labelsAllAllIndicestimeText[j].keys():
                    labelsAllAllIndicestimeText[j][k] += v
                else:
                    labelsAllAllIndicestimeText[j][k] = v
        for j in range(0, 2):
            for k, v in labelsTimeBins[j].items():
                if k in labelsAllAllTime[j].keys():
                    labelsAllAllTime[j][k] += v
                else:
                    labelsAllAllTime[j][k] = v

    file = open("labelsAllAllTimeBERTEverything", "wb")
    pickle.dump(labelsAllAllTime, file)
    file.close()
    file = open("labelsBinsDomainTimeBERTEverything", "wb")
    pickle.dump(labelsBinsDomainTime, file)
    file.close()
    file = open("labelsDomainAlltimeTextBERTEverything", "wb")
    pickle.dump(labelsDomainAlltimeText, file)
    file.close()
    file=open("labelsAllAlltimeTextBERTEverything", "wb")
    pickle.dump(labelsAllAlltimeText, file)
    file.close()
    file = open("labelsDomainAllIndicestimeTextBERTEverything", "wb")
    pickle.dump(labelsDomainAllIndicestimeText, file)
    file.close()
    file=open("labelsAllAllIndicestimeTextBERTEverything", "wb")
    pickle.dump(labelsAllAllIndicestimeText, file)
    file.close()
    file = open("domainsBERTEverything", "wb")
    pickle.dump(list(domains), file)
    file.close()
    print("Occurence time timeText")
    print("Domain")
    print(timesDomainOccurenceA)
    print("All")
    print(timesAllOccurenceA)
    file = open("timesDomainOccurenceABERTEverything", "wb")
    pickle.dump(timesDomainOccurenceA, file)
    file.close()
    file = open("timesAllOccurenceABERTEverything", "wb")
    pickle.dump(timesAllOccurenceA, file)
    file.close()
    print("Time")
    SpearmanLabelsAllIntra = getIntraRankingLabelsAll(labelsAllAllTime)
    meansAllIntra, stdAllIntra = calculateMeanAndStdAll(SpearmanLabelsAllIntra)
    print("Intra")
    print(meansAllIntra)
    print(stdAllIntra)
    file = open("Time-Intra-meansTEBERTEverything", "wb")
    pickle.dump(meansAllIntra, file)
    file.close()
    file = open("Time-Intra-stdTEBERTEverything", "wb")
    pickle.dump(stdAllIntra, file)
    file.close()
    print("Inter")
    SpearmanLabelsAllInter = getInterRankingLabelsAll(labelsAllAllTime)
    meansAllInter, stdAllInter = calculateMeanAndStdAll(SpearmanLabelsAllInter)
    print(meansAllInter)
    print(stdAllInter)
    file = open("Time-Inter-meansTEBERTEverything", "wb")
    pickle.dump(meansAllInter, file)
    file.close()
    file = open("Time-Inter-stdTEBERTEverything", "wb")
    pickle.dump(stdAllInter, file)
    file.close()
    print("Domain intra all")
    SpearmanLabelsDomainAllIntra, SpearmanLabelsDomainIntra = getIntraRankingLabelsDomain(labelsBinsDomainTime, domains)
    meansDomainAll, stdDomainAll = calculateMeanAndStdAll(SpearmanLabelsDomainAllIntra)
    print(meansDomainAll)
    print(stdDomainAll)
    file = open("Time-Intra-DomainAll-meansTEBERTEverything", "wb")
    pickle.dump(meansDomainAll, file)
    file.close()
    file = open("Time-Intra-DomainAll-stdTEBERTEverything", "wb")
    pickle.dump(stdDomainAll, file)
    file.close()
    print("Domain intra domain")
    meansDomain, stdDomain = calculateMeanAndStdDomain(SpearmanLabelsDomainIntra, domains)
    print(meansDomain)
    print(stdDomain)
    file = open("Time-Intra-Domain-meansTEBERTEverything", "wb")
    pickle.dump(meansDomain, file)
    file.close()
    file = open("Time-Intra-Domain-stdTEBERTEverything", "wb")
    pickle.dump(stdDomain, file)
    file.close()
    print("Domain inter all")
    SpearmanLabelsDomainAllInter, SpearmanLabelsDomainInter = getInterRankingLabelsDomain(labelsBinsDomainTime, domains)
    meansDomainAllInter, stdDomainAllInter = calculateMeanAndStdAll(SpearmanLabelsDomainAllInter)
    print(meansDomainAllInter)
    print(stdDomainAllInter)
    file = open("Time-Inter-DomainAll-meansTEBERTEverything", "wb")
    pickle.dump(meansDomainAllInter, file)
    file.close()
    file = open("Time-Inter-DomainAll-stdTEBERTEverything", "wb")
    pickle.dump(stdDomainAllInter, file)
    file.close()
    print("Domain inter domain")
    meansDomain, stdDomain = calculateMeanAndStdDomain(SpearmanLabelsDomainInter, domains)
    print(meansDomain)
    print(stdDomain)
    file = open("Time-Inter-Domain-meansTEBERTEverything", "wb")
    pickle.dump(meansDomain, file)
    file.close()
    file = open("Time-Inter-Domain-stdTEBERTEverything", "wb")
    pickle.dump(stdDomain, file)
    file.close()
    print("timeText version")
    SpearmanLabelsAllIntra = getIntraRankingLabelsAlltimeText(labelsAllAlltimeText)
    meansAllIntra,stdAllIntra = calculateMeanAndStdAlltimeText(SpearmanLabelsAllIntra)
    print("Intra")
    print(meansAllIntra)
    print(stdAllIntra)
    file = open("Time-Intra-meansAEBERTEverything", "wb")
    pickle.dump(meansAllIntra, file)
    file.close()
    file = open("Time-Intra-stdAEBERTEverything", "wb")
    pickle.dump(stdAllIntra, file)
    file.close()
    print("Inter")
    SpearmanLabelsAllInter = getInterRankingLabelsAlltimeText(labelsAllAlltimeText,labelsAllAllIndicestimeText)
    meansAllInter, stdAllInter = calculateMeanAndStdAlltimeText(SpearmanLabelsAllInter)
    print(meansAllInter)
    print(stdAllInter)
    file = open("Time-Inter-meansAEBERTEverything", "wb")
    pickle.dump(meansAllInter, file)
    file.close()
    file = open("Time-Inter-stdAEBERTEverything", "wb")
    pickle.dump(stdAllInter, file)
    file.close()
    print("Domain intra all")
    SpearmanLabelsDomainAllIntra,SpearmanLabelsDomainIntra = getIntraRankingLabelsDomaintimeText(labelsDomainAlltimeText,domains)
    meansDomainAll, stdDomainAll = calculateMeanAndStdAlltimeText(SpearmanLabelsDomainAllIntra)
    print(meansDomainAll)
    print(stdDomainAll)
    file = open("Time-Intra-DomainAll-meansAEBERTEverything", "wb")
    pickle.dump(meansDomainAll, file)
    file.close()
    file = open("Time-Intra-DomainAll-stdAEBERTEverything", "wb")
    pickle.dump(stdDomainAll, file)
    file.close()
    print("Domain intra domain")
    meansDomain, stdDomain = calculateMeanAndStdDomaintimeText(SpearmanLabelsDomainIntra,domains)
    print(meansDomain)
    print(stdDomain)
    file = open("Time-Intra-Domain-meansAEBERTEverything", "wb")
    pickle.dump(meansDomain, file)
    file.close()
    file = open("Time-Intra-Domain-stdAEBERTEverything", "wb")
    pickle.dump(stdDomain, file)
    file.close()
    print("Domain inter all")
    SpearmanLabelsDomainAllInter, SpearmanLabelsDomainInter = getInterRankingLabelsDomaintimeText(labelsDomainAlltimeText, domains,labelsDomainAllIndicestimeText)
    meansDomainAllInter, stdDomainAllInter = calculateMeanAndStdAlltimeText(SpearmanLabelsDomainAllInter)
    print(meansDomainAllInter)
    print(stdDomainAllInter)
    file = open("Time-Inter-DomainAll-meansAEBERTEverything", "wb")
    pickle.dump(meansDomainAllInter, file)
    file.close()
    file = open("Time-Inter-DomainAll-stdAEBERTEverything", "wb")
    pickle.dump(stdDomainAllInter, file)
    file.close()
    print("Domain inter domain")
    meansDomain, stdDomain = calculateMeanAndStdDomaintimeText(SpearmanLabelsDomainInter, domains)
    print(meansDomain)
    print(stdDomain)
    file = open("Time-Inter-Domain-meansAEBERTEverything", "wb")
    pickle.dump(meansDomain, file)
    file.close()
    file = open("Time-Inter-Domain-stdAEBERTEverything", "wb")
    pickle.dump(stdDomain, file)
    file.close()